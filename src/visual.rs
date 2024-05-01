use std::collections::VecDeque;

use miniquad::*;
use ndarray::Axis;

use crate::RungeKuttaIntegrator;

#[repr(C)]
struct Vec2 {
    x: f32,
    y: f32,
}

#[repr(C)]
struct Rgba {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Rgba {
    fn new() -> Self {
        Rgba {
            r: 0.0,
            b: 0.0,
            g: 0.0,
            a: 1.0,
        }
    }
}

#[repr(C)]
struct Point {
    pos: Vec2,
    color: Rgba,
}

pub struct StageConf {
    pub scale: f32,
    pub steps_per_frame: u32,
    pub trail_length: usize,
    pub nparticles: usize,
}

pub struct Stage {
    pipeline: Pipeline,
    bindings: Bindings,
    uniforms: shader::Uniforms,
    integrator: RungeKuttaIntegrator,
    conf: StageConf,
    positions: VecDeque<Point>,
    ctx: Box<dyn RenderingBackend>,
}

impl Stage {
    pub fn new(conf: StageConf, integrator: RungeKuttaIntegrator) -> Stage {
        let mut ctx: Box<dyn RenderingBackend> = window::new_rendering_backend();

        let s = 0.005; // size of bounding box
        let f = 1.0 / (2.0_f32).sqrt();

        #[rustfmt::skip]
        let vertices: [Vec2; 8] = [
            Vec2 { x:      s, y:    0.0 },
            Vec2 { x:  f * s, y:  f * s },
            Vec2 { x:    0.0, y:      s },
            Vec2 { x: -f * s, y:  f * s },
            Vec2 { x:     -s, y:    0.0 },
            Vec2 { x: -f * s, y: -f * s },
            Vec2 { x:    0.0, y:     -s },
            Vec2 { x:  f * s, y: -f * s },
        ];

        let geom_vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&vertices),
        );

        let indices: [u16; 24] = [
            0, 1, 4, 1, 2, 5, 2, 3, 6, 3, 4, 7, 4, 5, 0, 5, 6, 1, 6, 7, 2, 7, 0, 3,
        ];

        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        );

        let pos_vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Stream,
            BufferSource::empty::<Point>(conf.trail_length),
        );

        let bindings = Bindings {
            vertex_buffers: vec![geom_vertex_buffer, pos_vertex_buffer],
            index_buffer,
            images: vec![],
        };

        let shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: shader::VERTEX,
                    fragment: shader::FRAGMENT,
                },
                shader::meta(),
            )
            .unwrap();

        let pipeline = ctx.new_pipeline(
            &[
                BufferLayout::default(),
                BufferLayout {
                    step_func: VertexStep::PerInstance,
                    ..Default::default()
                },
            ],
            &[
                VertexAttribute::with_buffer("in_pos", VertexFormat::Float2, 0),
                VertexAttribute::with_buffer("in_inst_pos", VertexFormat::Float2, 1),
                VertexAttribute::with_buffer("in_inst_col", VertexFormat::Float4, 1),
            ],
            shader,
            PipelineParams::default(),
        );

        let uniforms = shader::Uniforms { blobs_count: 3 };

        let positions = VecDeque::with_capacity(conf.trail_length);

        Stage {
            pipeline,
            bindings,
            uniforms,
            integrator,
            conf,
            positions,
            ctx,
        }
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {
        for _ in 1..self.conf.steps_per_frame {
            self.integrator.next();
        }

        let state = self.integrator.next();

        if let Some((_t, pq)) = state {
            let p = pq.index_axis(Axis(0), 1);

            p.axis_chunks_iter(Axis(0), 2)
                .enumerate()
                .for_each(|(i, pos)| {
                    if self.positions.len() >= self.conf.nparticles {
                        let col = match i {
                            0 => Rgba {
                                r: 0.0,
                                g: 0.0,
                                b: 0.5,
                                a: 1.0,
                            },
                            1 => Rgba {
                                r: 0.0,
                                g: 0.5,
                                b: 0.0,
                                a: 1.0,
                            },
                            2 => Rgba {
                                r: 0.5,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            },
                            _ => Rgba {
                                r: 0.5,
                                g: 0.5,
                                b: 0.5,
                                a: 0.5,
                            },
                        };
                        self.positions
                            .get_mut(self.positions.len() - self.conf.nparticles)
                            .unwrap()
                            .color = col
                    }
                    if self.positions.len() >= self.conf.trail_length {
                        self.positions.pop_front();
                    }

                    self.positions.push_back(Point {
                        pos: Vec2 {
                            x: pos[0] as f32 / self.conf.scale, //  + 0.5,
                            y: pos[1] as f32 / self.conf.scale, //  + 0.5,
                        },
                        color: Rgba {
                            r: 0.75,
                            g: 1.0,
                            b: 0.50,
                            a: 1.0,
                        },
                    });
                });
        }
    }

    fn draw(&mut self) {
        self.ctx.buffer_update(
            self.bindings.vertex_buffers[1],
            BufferSource::slice(self.positions.make_contiguous()),
        );

        self.ctx.begin_default_pass(Default::default());

        self.ctx.apply_pipeline(&self.pipeline);
        self.ctx.apply_bindings(&self.bindings);
        self.ctx
            .apply_uniforms(UniformsSource::table(&self.uniforms));

        self.ctx.draw(0, 24, self.positions.len() as i32);

        self.ctx.end_render_pass();
        self.ctx.commit_frame();
    }
}

mod shader {
    use miniquad::{ShaderMeta, UniformBlockLayout, UniformDesc, UniformType};

    pub const VERTEX: &str = r#"#version 330 core
        attribute vec2 in_pos;
        attribute vec2 in_inst_pos;
        attribute vec4 in_inst_col;

        varying lowp vec4 color;

        void main() {
            gl_Position = vec4(in_pos + in_inst_pos, 0.0, 1.0);
            color = in_inst_col;
        }"#;

    pub const FRAGMENT: &str = r#"#version 330 core
        // precision highp float;

        varying lowp vec4 color;

        uniform int blobs_count;

        void main() {
            gl_FragColor = color;
        }

        // vec4 colors[8] = vec4[](
        //     vec4(0.75, 1.0, 0.50, 1.0),
        //     vec4(0.88, 0.88, 0.44, 1.0),
        //     vec4(0.88, 0.62, 0.22, 1.0),
        //     vec4(0.75, 0.50, 0.11, 1.0),
        //     vec4(0.75, 0.38, 0.00, 1.0),
        //     vec4(0.62, 0.25, 0.00, 1.0),
        //     vec4(0.50, 0.094, 0.00, 1.0),
        //     vec4(0.31, 0.0039, 0.062, 1.0)
        // );

        // vec2 coord;
        // vec4 color;

        // void main() {
        //     coord = uv;

        //     color = vec4(0.0, 0.0, 0.0, 1.0);
        //     for (int i = 0; i < 8; i++) {
        //         float d = distance(coord, blobs_positions[i]);
        //         if (d < 0.01) {
        //             color = colors[i];
        //         }
        //     }
        //     gl_FragColor = color;
        // }
"#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![UniformDesc::new("blobs_count", UniformType::Int1)],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub blobs_count: i32,
    }
}
