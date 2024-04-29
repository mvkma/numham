use std::collections::VecDeque;

use miniquad::*;

use crate::RungeKuttaIntegrator;

#[repr(C)]
struct Vec2 {
    x: f32,
    y: f32,
}

#[repr(C)]
struct Vertex {
    pos: Vec2,
    uv: Vec2,
}

pub struct StageConf {
    pub scale: f32,
    pub steps_per_frame: u32,
    pub trail_length: usize,
}

pub struct Stage {
    pipeline: Pipeline,
    bindings: Bindings,
    uniforms: shader::Uniforms,
    integrator: RungeKuttaIntegrator,
    conf: StageConf,
    positions: VecDeque<Vec2>,
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
            BufferSource::empty::<Vec2>(conf.trail_length),
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
            let positions = self.integrator.ham.positions(&pq);

            (0..positions.len()).for_each(|i| {
                // self.uniforms.blobs_positions[i].0 = positions[i][0] as f32 / self.conf.scale + 0.5;
                // self.uniforms.blobs_positions[i].1 = positions[i][1] as f32 / self.conf.scale + 0.5;
                if self.positions.len() >= self.conf.trail_length {
                    self.positions.pop_front();
                }

                self.positions.push_back(Vec2 {
                    x: positions[i][0] as f32 / self.conf.scale, //  + 0.5,
                    y: positions[i][1] as f32 / self.conf.scale, //  + 0.5,
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

        void main() {
            gl_Position = vec4(in_pos + in_inst_pos, 0.0, 1.0);
        }"#;

    pub const FRAGMENT: &str = r#"#version 330 core
        // precision highp float;

        // varying vec2 uv;

        uniform int blobs_count;

        void main() {
            gl_FragColor = vec4(0.75, 1.0, 0.50, 1.0);
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
