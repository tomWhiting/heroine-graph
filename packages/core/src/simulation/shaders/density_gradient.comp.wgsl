// Density Gradient Compute Shader
// Computes the gradient of a density field using Sobel operators
//
// The gradient points toward higher density. For repulsion,
// nodes will move in the OPPOSITE direction (away from density).

struct GradientUniforms {
    width: u32,
    height: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: GradientUniforms;
@group(0) @binding(1) var density_texture: texture_2d<f32>;
@group(0) @binding(2) var gradient_texture: texture_storage_2d<rg32float, write>;

// Sample density at a pixel coordinate with bounds checking
fn sample_density(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(uniforms.width) - 1);
    let cy = clamp(y, 0, i32(uniforms.height) - 1);
    return textureLoad(density_texture, vec2<i32>(cx, cy), 0).r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (global_id.x >= uniforms.width || global_id.y >= uniforms.height) {
        return;
    }

    // Sobel operators for gradient computation
    // Gx kernel:       Gy kernel:
    // -1  0  1         -1 -2 -1
    // -2  0  2          0  0  0
    // -1  0  1          1  2  1

    // Sample 3x3 neighborhood
    let tl = sample_density(x - 1, y - 1);
    let tc = sample_density(x,     y - 1);
    let tr = sample_density(x + 1, y - 1);
    let ml = sample_density(x - 1, y);
    // let mc = sample_density(x,     y);  // center not needed for Sobel
    let mr = sample_density(x + 1, y);
    let bl = sample_density(x - 1, y + 1);
    let bc = sample_density(x,     y + 1);
    let br = sample_density(x + 1, y + 1);

    // Compute gradient using Sobel
    let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);

    // Store gradient (pointing toward higher density)
    // RG channels store X and Y components
    textureStore(gradient_texture, vec2<i32>(x, y), vec4<f32>(gx, gy, 0.0, 0.0));
}
