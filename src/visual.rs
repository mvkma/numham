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
}

pub struct Stage {
    pipeline: Pipeline,
    bindings: Bindings,
    uniforms: shader::Uniforms,
    integrator: RungeKuttaIntegrator,
    conf: StageConf,
    ctx: Box<dyn RenderingBackend>,
}

impl Stage {
    pub fn new(conf: StageConf, integrator: RungeKuttaIntegrator) -> Stage {
        let mut ctx: Box<dyn RenderingBackend> = window::new_rendering_backend();

        let s = 1.0; // size of bounding box

        let vertices: [Vertex; 4] = [
            Vertex {
                pos: Vec2 { x: s, y: s },
                uv: Vec2 { x: 1.0, y: 1.0 },
            },
            Vertex {
                pos: Vec2 { x: s, y: -s },
                uv: Vec2 { x: 1.0, y: 0.0 },
            },
            Vertex {
                pos: Vec2 { x: -s, y: -s },
                uv: Vec2 { x: 0.0, y: 0.0 },
            },
            Vertex {
                pos: Vec2 { x: -s, y: s },
                uv: Vec2 { x: 0.0, y: 1.0 },
            },
        ];

        let vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&vertices),
        );

        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];

        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        );

        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
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
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("in_pos", VertexFormat::Float2),
                VertexAttribute::new("in_uv", VertexFormat::Float2),
            ],
            shader,
            PipelineParams::default(),
        );

        let uniforms = shader::Uniforms {
            blobs_count: 8,
            blobs_positions: [
                (-10.0, -10.0),
                (0.5, 0.5),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ],
        };

        Stage {
            pipeline,
            bindings,
            uniforms,
            integrator,
            conf,
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
                self.uniforms.blobs_positions[i].0 = positions[i][0] as f32 / self.conf.scale + 0.5;
                self.uniforms.blobs_positions[i].1 = positions[i][1] as f32 / self.conf.scale + 0.5;
            });
        }
    }

    fn draw(&mut self) {
        self.ctx.begin_default_pass(Default::default());

        self.ctx.apply_pipeline(&self.pipeline);
        self.ctx.apply_bindings(&self.bindings);
        self.ctx
            .apply_uniforms(UniformsSource::table(&self.uniforms));

        self.ctx.draw(0, 6, 1);

        self.ctx.end_render_pass();
        self.ctx.commit_frame();
    }
}

mod shader {
    use miniquad::{ShaderMeta, UniformBlockLayout, UniformDesc, UniformType};

    pub const VERTEX: &str = r#"#version 330 core
        attribute vec2 in_pos;
        attribute vec2 in_uv;

        varying highp vec2 uv;

        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            uv = in_uv;
        }"#;

    pub const FRAGMENT: &str = r#"#version 330 core
        precision highp float;

        varying vec2 uv;

        uniform int blobs_count;
        uniform vec2 blobs_positions[8];

        vec4 colors[8] = vec4[](
            vec4(0.75, 1.0, 0.50, 1.0),
            vec4(0.88, 0.88, 0.44, 1.0),
            vec4(0.88, 0.62, 0.22, 1.0),
            vec4(0.75, 0.50, 0.11, 1.0),
            vec4(0.75, 0.38, 0.00, 1.0),
            vec4(0.62, 0.25, 0.00, 1.0),
            vec4(0.50, 0.094, 0.00, 1.0),
            vec4(0.31, 0.0039, 0.062, 1.0)
        );

        vec2 coord;
        vec4 color;

        void main() {
            coord = uv;

            color = vec4(0.0, 0.0, 0.0, 1.0);
            for (int i = 0; i < 8; i++) {
                float d = distance(coord, blobs_positions[i]);
                if (d < 0.01) {
                    color = colors[i];
                }
            }
            gl_FragColor = color;
        }"#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("blobs_count", UniformType::Int1),
                    UniformDesc::new("blobs_positions", UniformType::Float2).array(8),
                ],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub blobs_count: i32,
        pub blobs_positions: [(f32, f32); 8],
    }
}
