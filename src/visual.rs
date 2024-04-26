use std::f64::consts;

use miniquad::*;

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

pub struct Stage {
    pipeline: Pipeline,
    bindings: Bindings,
    uniforms: shader::Uniforms,
    ctx: Box<dyn RenderingBackend>,
}

impl Stage {
    pub fn new() -> Stage {
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
            blobs_positions: [(0.0, 0.0); 8],
        };

        Stage {
            pipeline,
            bindings,
            uniforms,
            ctx,
        }
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {
        let t = date::now();

        for i in 0..self.uniforms.blobs_count as usize {
            // let t = t + i as f64 * 0.3;

            let phi = consts::PI / 3.0 * i as f64;
            self.uniforms.blobs_positions[i].0 = 0.5 + ((3.0 * t + phi).sin() as f32) * 0.5;
            self.uniforms.blobs_positions[i].1 = 0.5 + ((1.0 * t + phi).cos() as f32) * 0.5;
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

        vec2 coord;
        vec4 color;

        void main() {
            coord = uv;

            color = vec4(0.0, 0.0, 0.0, 1.0);
            for (int i = 0; i < 8; i++) {
                float d = distance(coord, blobs_positions[i]);
                if (d < 0.1) {
                    color = vec4(1.0, 0.0, 0.0, 1.0);
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
