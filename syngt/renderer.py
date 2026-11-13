"""Headless OpenGL renderer (moderngl).

Passes per camera:
  * colour   : PBR (GGX) shading, N lights, hemisphere ambient, 1 shadow-casting light, MSAA, linear HDR RGBA
  * gbuffer  : instance id, metric z-depth, camera-frame normals (exact, no AA)
  * solo     : per-object silhouette without occluders (amodal mask) - read back only in its bbox
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import moderngl
import numpy as np

from .assets import MeshAsset
from .geometry import (gl_look_at, gl_ortho, gl_perspective, gl_projection_from_K, gl_view_from_cv, inv_T,
                       transform_points)

MAX_LIGHTS = 8

_VS = """
#version 330
in vec3 in_pos; in vec3 in_normal; in vec2 in_uv; in vec4 in_color;
uniform mat4 M; uniform mat4 V; uniform mat4 P; uniform mat4 light_vp;
out vec3 v_world; out vec3 v_normal; out vec2 v_uv; out vec4 v_color; out vec4 v_ls; out vec3 v_view;
void main(){
  vec4 w = M * vec4(in_pos, 1.0);
  v_world = w.xyz;
  v_normal = mat3(M) * in_normal;   // rigid transforms + uniform scale only
  v_uv = in_uv; v_color = in_color;
  v_ls = light_vp * w;
  vec4 vv = V * w; v_view = vv.xyz;
  gl_Position = P * vv;
}
"""

_FS_PBR = """
#version 330
in vec3 v_world; in vec3 v_normal; in vec2 v_uv; in vec4 v_color; in vec4 v_ls; in vec3 v_view;
uniform vec3 cam_pos;
uniform vec4 base_color; uniform int use_texture; uniform int use_vcolor; uniform sampler2D tex;
uniform vec3 tint;
uniform float roughness; uniform float metallic;
uniform int uv_mode; uniform mat2 uv_rot; uniform vec2 uv_scale;
uniform int n_lights; uniform vec4 light_pos[8]; uniform vec3 light_col[8];
uniform vec3 amb_sky; uniform vec3 amb_ground;
uniform int shadow_on; uniform sampler2DShadow shadow_map; uniform float shadow_texel; uniform float shadow_bias;
uniform float shadow_radius; uniform int catcher; uniform float catcher_strength;
out vec4 frag;
const float PI = 3.14159265;
float D_ggx(float nh, float a){ float a2=a*a; float d=nh*nh*(a2-1.0)+1.0; return a2/(PI*d*d+1e-7); }
float G_s(float nx, float k){ return nx/(nx*(1.0-k)+k); }
vec3 F_s(float c, vec3 F0){ return F0+(1.0-F0)*pow(1.0-c,5.0); }
float shadow_factor(vec3 n, vec3 l){
  if(shadow_on==0) return 1.0;
  vec3 p = v_ls.xyz / v_ls.w; p = p*0.5+0.5;
  if(p.x<0.0||p.x>1.0||p.y<0.0||p.y>1.0||p.z>1.0) return 1.0;
  float bias = shadow_bias*(1.0+3.0*(1.0-max(dot(n,l),0.0)));
  float s=0.0;
  float r = max(shadow_radius, 0.5) * 0.5;      // 5x5 PCF, kernel scaled by the softness (penumbra)
  for(int i=-2;i<=2;i++) for(int j=-2;j<=2;j++)
    s += texture(shadow_map, vec3(p.xy+vec2(i,j)*shadow_texel*r, p.z-bias*(1.0+0.25*r)));
  return s/25.0;
}
void main(){
  vec3 n = normalize(v_normal);
  if(!gl_FrontFacing) n = -n;
  if(catcher==1){                                 // invisible ground that only receives the key-light shadow
    vec3 L0 = light_pos[0].w < 0.5 ? normalize(light_pos[0].xyz) : normalize(light_pos[0].xyz - v_world);
    float sh = (n_lights > 0) ? shadow_factor(n, L0) : 1.0;
    frag = vec4(0.0, 0.0, 0.0, catcher_strength * (1.0 - sh));
    return;
  }
  vec2 uv = v_uv;
  if(uv_mode==1) uv = (uv_rot * v_world.xy) / uv_scale;
  vec3 albedo = base_color.rgb;
  if(use_texture==1){ vec4 t = texture(tex, uv); albedo *= pow(t.rgb, vec3(2.2)); }
  if(use_vcolor==1){ albedo *= pow(v_color.rgb, vec3(2.2)); }
  albedo *= tint;
  float r = clamp(roughness, 0.04, 1.0); float a = r*r;
  vec3 V = normalize(cam_pos - v_world);
  vec3 F0 = mix(vec3(0.04), albedo, metallic);
  vec3 Lo = vec3(0.0);
  float nv = max(dot(n,V), 1e-4);
  float k = (r+1.0)*(r+1.0)/8.0;
  for(int i=0;i<n_lights;i++){
    vec3 L; vec3 rad;
    if(light_pos[i].w < 0.5){ L = normalize(light_pos[i].xyz); rad = light_col[i]; }
    else { vec3 d = light_pos[i].xyz - v_world; float dist = length(d); L = d/dist; rad = light_col[i]/(dist*dist); }
    float nl = max(dot(n,L),0.0);
    if(nl<=0.0) continue;
    vec3 H = normalize(V+L);
    float nh = max(dot(n,H),0.0);
    vec3 F = F_s(max(dot(H,V),0.0), F0);
    float D = D_ggx(nh, a); float G = G_s(nl,k)*G_s(nv,k);
    vec3 spec = D*G*F/(4.0*nl*nv+1e-4);
    vec3 kd = (1.0-F)*(1.0-metallic);
    float sh = (i==0) ? shadow_factor(n,L) : 1.0;
    Lo += (kd*albedo/PI + spec) * rad * nl * sh;
  }
  float hemi = n.z*0.5+0.5;
  vec3 amb = mix(amb_ground, amb_sky, hemi);
  vec3 Fa = F0 + (max(vec3(1.0-r),F0)-F0)*pow(1.0-nv,5.0);
  vec3 refl = reflect(-V, n);
  vec3 env_spec = mix(amb_ground, amb_sky, refl.z*0.5+0.5);
  vec3 ambient = amb*albedo*(1.0-metallic)*(1.0-Fa) + env_spec*Fa*(1.0-0.6*r);
  frag = vec4(Lo + ambient, 1.0);
}
"""

_FS_GBUF = """
#version 330
in vec3 v_world; in vec3 v_normal; in vec2 v_uv; in vec4 v_color; in vec4 v_ls; in vec3 v_view;
uniform float inst_id; uniform mat3 V3;
layout(location=0) out float o_id;
layout(location=1) out float o_depth;
layout(location=2) out vec4 o_normal;
void main(){
  vec3 n = normalize(V3 * normalize(v_normal));   // GL camera frame
  if(!gl_FrontFacing) n = -n;
  o_id = inst_id;
  o_depth = -v_view.z;
  o_normal = vec4(n.x, -n.y, -n.z, 1.0);             // -> OpenCV camera frame
}
"""

_FS_SOLO = """
#version 330
out float o;
void main(){ o = 1.0; }
"""

_FS_SHADOW = """
#version 330
void main(){}
"""

# ---------------------------------------------------------------- GPU sensor post-processing
_VS_FULL = """
#version 330
void main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }
"""

_FS_COMPOSE = """
#version 330
uniform sampler2D canvas; uniform sampler2D map; uniform sampler2D bg;
uniform int distorted; uniform ivec2 offset; uniform vec2 canvas_size; uniform vec2 out_size;
uniform int has_bg; uniform float bg_gain; uniform vec3 gain; uniform float vignette;
out vec4 o;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  int yi = int(out_size.y) - 1 - p.y;                         // image row (top-down)
  vec4 c;
  if(distorted == 1){
    vec2 m = texelFetch(map, p, 0).xy;                         // canvas pixel coords (top-down)
    if(m.x < -0.5 || m.y < -0.5 || m.x > canvas_size.x - 0.5 || m.y > canvas_size.y - 0.5) c = vec4(0.0);
    else c = texture(canvas, vec2((m.x + 0.5) / canvas_size.x, 1.0 - (m.y + 0.5) / canvas_size.y));
  } else {
    c = texelFetch(canvas, ivec2(p.x + offset.x, int(canvas_size.y) - 1 - (yi + offset.y)), 0);
  }
  float a = clamp(c.a, 0.0, 1.0);
  vec3 rgb = c.rgb;
  if(has_bg == 1){
    vec3 b = texelFetch(bg, p, 0).rgb;
    rgb += pow(clamp(b * bg_gain, 0.0, 1.0), vec3(2.2)) * (1.0 - a);
  }
  rgb *= gain;
  if(vignette > 0.0){
    float x = (float(p.x) - out_size.x / 2.0) / (out_size.x / 2.0);
    float y = (float(yi) - out_size.y / 2.0) / (out_size.y / 2.0);
    rgb *= 1.0 - vignette * (x * x + y * y) / 2.0;
  }
  o = vec4(rgb, a);
}
"""

_FS_GAUSS = """
#version 330
uniform sampler2D src; uniform ivec2 dir; uniform float sigma;
out vec4 o;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy); ivec2 sz = textureSize(src, 0);
  int r = int(ceil(4.0 * sigma));
  vec4 acc = vec4(0.0); float ws = 0.0;
  for(int i = -r; i <= r; i++){
    float w = exp(-0.5 * float(i * i) / (sigma * sigma));
    acc += w * texelFetch(src, clamp(p + dir * i, ivec2(0), sz - 1), 0); ws += w;
  }
  o = acc / ws;
}
"""

_FS_MOTION = """
#version 330
uniform sampler2D src; uniform vec2 dir; uniform int L;
out vec4 o;
void main(){
  vec2 sz = vec2(textureSize(src, 0)); vec4 acc = vec4(0.0);
  for(int i = 0; i < L; i++){ float t = float(i) - 0.5 * float(L - 1); acc += texture(src, (gl_FragCoord.xy + dir * t) / sz); }
  o = acc / float(L);
}
"""

_FS_FINAL = """
#version 330
uniform sampler2D src; uniform float shot; uniform float readn; uniform uint seed;
uniform int filmic; uniform float sat; uniform float con; uniform int keep_alpha;
uniform int post_bg; uniform sampler2D bg; uniform float bg_gain;   // real photo: composited after the sensor
uniform float ae_gain;                                                  // auto exposure (1 = off)
out vec4 o;
uint pcg(uint v){ uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float u01(inout uint s){ s = pcg(s); return (float(s) + 0.5) / 4294967296.0; }
float gauss(inout uint s){ float a = u01(s); float b = u01(s); return sqrt(-2.0 * log(a)) * cos(6.28318531 * b); }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 c = texelFetch(src, p, 0); vec3 x = c.rgb;
  float a = clamp(c.a, 0.0, 1.0);
  if(post_bg == 1 || keep_alpha == 1) x = (a > 1e-4) ? x / a : vec3(0.0);   // un-premultiply the rendered layer
  x *= ae_gain;
  uint s = pcg(uint(p.x) ^ pcg(uint(p.y) ^ pcg(seed)));
  if(shot > 0.0) x += vec3(gauss(s), gauss(s), gauss(s)) * sqrt(max(x, vec3(0.0))) * shot;
  if(readn > 0.0) x += vec3(gauss(s), gauss(s), gauss(s)) * readn;
  x = max(x, vec3(0.0));
  if(filmic == 1) x = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
  else x = clamp(x, 0.0, 1.0);
  x = pow(x, vec3(1.0 / 2.2));
  float g = (x.r + x.g + x.b) / 3.0;
  x = g + (x - g) * sat;
  x = 0.5 + (x - 0.5) * con;
  x = clamp(x, 0.0, 1.0);
  if(post_bg == 1){ x = x * a + clamp(texelFetch(bg, p, 0).rgb * bg_gain, 0.0, 1.0) * (1.0 - a); a = 1.0; }
  o = vec4(x, keep_alpha == 1 ? a : 1.0);
}
"""


@dataclass
class Instance:
    asset: MeshAsset
    T_world: np.ndarray
    inst_id: int
    annotate: bool = True
    base_color: Optional[Tuple[float, float, float]] = None   # linear rgb override; None -> asset colours
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)       # multiplicative (e.g. hue jitter of textures)
    roughness: float = 0.5
    metallic: float = 0.0
    use_texture: bool = True
    scale: float = 1.0                                         # uniform scale applied in object frame


@dataclass
class Light:
    kind: str                       # "point" | "directional"
    vec: np.ndarray                 # position (point) or direction *towards* light (directional)
    color: np.ndarray               # linear rgb * intensity


@dataclass
class Ground:
    enabled: bool = True
    visible: bool = True
    texture: Optional[np.ndarray] = None   # uint8 HxWx3
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    tile_size: float = 0.5
    tile_size_y: float = 0.0               # 0 -> square tiles
    rotation_deg: float = 0.0
    roughness: float = 0.8
    extent: float = 6.0
    key: str = ""                           # cache key for texture upload
    catcher: bool = False                   # shadow catcher: invisible, only darkens the background photo
    catcher_strength: float = 0.7


@dataclass
class RenderScene:
    instances: List[Instance]
    ground: Ground
    lights: List[Light]
    ambient_sky: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.22]))
    ambient_ground: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.08, 0.08]))
    shadows: bool = True
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.5
    shadow_softness: float = 1.0


@dataclass
class RenderOutput:
    rgba: Optional[np.ndarray]       # float32 HxWx4 linear (premultiplied alpha); None if not read back
    inst: np.ndarray                 # int32 HxW instance id (0 = none)
    depth: np.ndarray                # float32 HxW metres (0 = none)
    normals: Optional[np.ndarray]    # float32 HxWx3 (OpenCV cam frame)
    solo: Dict[int, Tuple[np.ndarray, Tuple[int, int]]]  # inst_id -> (bool mask crop, (x0,y0))


class _GPUPart:
    def __init__(self, vao_pbr, vao_gb, vao_sh, vao_solo, texture, has_uv, has_color, base_color, buffers):
        self.vao_pbr, self.vao_gb, self.vao_sh, self.vao_solo = vao_pbr, vao_gb, vao_sh, vao_solo
        self.texture, self.has_uv, self.has_color, self.base_color = texture, has_uv, has_color, base_color
        self.buffers = buffers


def _srgb_to_lin(c):
    c = np.asarray(c, dtype=np.float64)
    return np.power(np.clip(c, 0, 1), 2.2)


def _create_context() -> "moderngl.Context":
    """Headless GPU context. EGL needs no X display (works over SSH, with a locked screen and with many worker
    processes); GLX/WGL is the fallback (Windows, drivers without EGL). SYNGT_GL_BACKEND=egl|default forces one."""
    import os
    want = os.environ.get("SYNGT_GL_BACKEND", "").lower()
    errors = []
    for backend in (["egl"] if want == "egl" else [] if want == "default" else ["egl"]) + [None]:
        try:
            if backend:
                return moderngl.create_standalone_context(require=330, backend=backend)
            return moderngl.create_standalone_context(require=330)
        except Exception as e:  # noqa: PERF203
            errors.append(f"{backend or 'default'}: {e}")
    raise RuntimeError("no OpenGL 3.3 context: " + " | ".join(errors))


class Renderer:
    def __init__(self, msaa: int = 4, shadow_size: int = 2048):
        self.ctx = _create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.msaa = int(max(0, min(msaa, self.ctx.max_samples)))
        self.prog_pbr = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS_PBR)
        self.prog_gb = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS_GBUF)
        self.prog_solo = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS_SOLO)
        self.prog_sh = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS_SHADOW)
        self.pp = {n: self.ctx.program(vertex_shader=_VS_FULL, fragment_shader=fs) for n, fs in
                   (("compose", _FS_COMPOSE), ("gauss", _FS_GAUSS), ("motion", _FS_MOTION), ("final", _FS_FINAL))}
        self.pp_vao = {n: self.ctx.vertex_array(p, []) for n, p in self.pp.items()}
        self._pp_fbos: Dict[Tuple[int, int], dict] = {}
        self._map_tex: Dict[int, tuple] = {}
        self.last_color = None
        self.last_size = (0, 0)
        self.meshes: Dict[str, List[_GPUPart]] = {}
        self.ground_textures: Dict[str, moderngl.Texture] = {}
        self._fbos: Dict[Tuple[int, int], dict] = {}
        self.shadow_size = shadow_size
        self.shadow_tex = self.ctx.depth_texture((shadow_size, shadow_size))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = self.ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.white = self.ctx.texture((1, 1), 4, bytes([255, 255, 255, 255]))
        self._ground_part = None
        self.info = dict(self.ctx.info)

    # ---------------------------------------------------------------- resources
    MAX_GROUND_TEXTURES = 6     # up to 4096^2 with mipmaps (~70 MB each) per process: keep only a few

    def release_unused(self, keep_keys: set) -> None:
        """Free every uploaded mesh that is not in keep_keys (models that were removed, one-off bins ...)."""
        for k in list(self.meshes):
            if k not in keep_keys:
                for p in self.meshes.pop(k):
                    for b in p.buffers:
                        b.release()
                    for v in (p.vao_pbr, p.vao_gb, p.vao_sh, p.vao_solo):
                        v.release()
                    if p.texture is not None and p.texture is not self.white:
                        p.texture.release()
        self._trim_ground_textures()

    def _trim_ground_textures(self, keep: str = "") -> None:
        for k in list(self.ground_textures)[:-self.MAX_GROUND_TEXTURES]:
            if k != keep:
                self.ground_textures.pop(k).release()

    def _upload(self, positions, normals, indices, uvs=None, colors=None, texture=None, base_color=(0.7, 0.7, 0.7, 1)):
        n = len(positions)
        uv = uvs if uvs is not None else np.zeros((n, 2), np.float32)
        col = colors if colors is not None else np.ones((n, 4), np.float32)
        if col.shape[1] == 3:
            col = np.hstack([col, np.ones((n, 1), np.float32)])
        data = np.hstack([positions.astype(np.float32), normals.astype(np.float32), uv.astype(np.float32),
                          col.astype(np.float32)]).astype(np.float32)
        vbo = self.ctx.buffer(data.tobytes())
        ibo = self.ctx.buffer(np.ascontiguousarray(indices, dtype=np.uint32).tobytes())

        def mk(prog):
            # skip attributes optimised away by the GLSL compiler (padding "Nx4")
            fmt_parts, names = [], []
            for a, n_comp in (("in_pos", 3), ("in_normal", 3), ("in_uv", 2), ("in_color", 4)):
                if a in prog:
                    fmt_parts.append(f"{n_comp}f")
                    names.append(a)
                else:
                    fmt_parts.append(f"{n_comp}x4")
            return self.ctx.vertex_array(prog, [(vbo, " ".join(fmt_parts), *names)],
                                         index_buffer=ibo, index_element_size=4)

        tex = None
        if texture is not None:
            img = np.ascontiguousarray(texture[::-1])
            comps = img.shape[2] if img.ndim == 3 else 1
            tex = self.ctx.texture((img.shape[1], img.shape[0]), comps, img.tobytes())
            tex.build_mipmaps()
            tex.anisotropy = 8.0
            tex.repeat_x = tex.repeat_y = True
        return _GPUPart(mk(self.prog_pbr), mk(self.prog_gb), mk(self.prog_sh), mk(self.prog_solo), tex,
                        uvs is not None, colors is not None, tuple(base_color), [vbo, ibo])

    def get_mesh(self, asset: MeshAsset) -> List[_GPUPart]:
        if asset.key not in self.meshes:
            self.meshes[asset.key] = [self._upload(p.positions, p.normals, p.indices, p.uvs, p.colors, p.texture,
                                                   p.base_color) for p in asset.parts]
        return self.meshes[asset.key]

    def _ground(self, extent: float) -> _GPUPart:
        if self._ground_part is None or self._ground_extent != extent:
            e = extent
            pos = np.array([[-e, -e, 0], [e, -e, 0], [e, e, 0], [-e, e, 0]], np.float32)
            nrm = np.tile(np.array([[0, 0, 1]], np.float32), (4, 1))
            idx = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
            self._ground_part = self._upload(pos, nrm, idx)
            self._ground_extent = extent
        return self._ground_part

    def _ground_texture(self, g: Ground):
        if g.texture is None:
            return None
        if g.key not in self.ground_textures:
            img = np.ascontiguousarray(g.texture[..., :3][::-1])
            t = self.ctx.texture((img.shape[1], img.shape[0]), 3, img.tobytes())
            t.build_mipmaps()
            t.anisotropy = 16.0
            t.repeat_x = t.repeat_y = True
            self.ground_textures[g.key] = t
            self._trim_ground_textures(keep=g.key)      # bounded even if the caller never calls release_unused
        return self.ground_textures[g.key]

    def _fbo(self, w: int, h: int) -> dict:
        key = (w, h)
        if key not in self._fbos:
            if len(self._fbos) > 6:
                for k in list(self._fbos)[:3]:
                    for o in self._fbos.pop(k).values():
                        o.release()
            d = {}
            if self.msaa > 1:
                d["ms_col"] = self.ctx.renderbuffer((w, h), 4, samples=self.msaa, dtype="f2")
                d["ms_dep"] = self.ctx.depth_renderbuffer((w, h), samples=self.msaa)
                d["ms_fbo"] = self.ctx.framebuffer([d["ms_col"]], d["ms_dep"])
            d["col"] = self.ctx.texture((w, h), 4, dtype="f2")  # same format as MSAA buffer (resolve rule)
            d["col"].filter = (moderngl.LINEAR, moderngl.LINEAR)
            d["col"].repeat_x = d["col"].repeat_y = False
            d["col_dep"] = self.ctx.depth_renderbuffer((w, h))
            d["col_fbo"] = self.ctx.framebuffer([d["col"]], d["col_dep"])
            d["id"] = self.ctx.renderbuffer((w, h), 1, dtype="f4")
            d["z"] = self.ctx.renderbuffer((w, h), 1, dtype="f4")
            d["n"] = self.ctx.renderbuffer((w, h), 4, dtype="f2")
            d["gb_dep"] = self.ctx.depth_renderbuffer((w, h))
            d["gb_fbo"] = self.ctx.framebuffer([d["id"], d["z"], d["n"]], d["gb_dep"])
            d["solo"] = self.ctx.renderbuffer((w, h), 1, dtype="f1")
            d["solo_dep"] = self.ctx.depth_renderbuffer((w, h))
            d["solo_fbo"] = self.ctx.framebuffer([d["solo"]], d["solo_dep"])
            self._fbos[key] = d
        return self._fbos[key]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _w(prog, name, value):
        if name in prog:
            prog[name].value = value

    @staticmethod
    def _mat(prog, name, M):
        if name in prog:
            prog[name].write(np.asarray(M, dtype=np.float32).T.tobytes())

    def _model_matrix(self, inst: Instance) -> np.ndarray:
        if inst.scale != 1.0:
            S = np.diag([inst.scale, inst.scale, inst.scale, 1.0])
            return inst.T_world @ S
        return inst.T_world

    def _shadow_matrix(self, scene: RenderScene) -> Optional[np.ndarray]:
        if not scene.shadows or not scene.lights:
            return None
        L = scene.lights[0]
        c, r = scene.center, max(scene.radius, 0.05) * 1.3
        if L.kind == "directional":
            d = np.asarray(L.vec, float)
            d = d / np.linalg.norm(d)
            eye = c + d * (r * 4 + 1.0)
            V = gl_look_at(eye, c)
            P = gl_ortho(-r, r, -r, r, 0.01, r * 8 + 2.0)
        else:
            eye = np.asarray(L.vec, float)
            dist = np.linalg.norm(eye - c)
            fov = math.degrees(2 * math.asin(min(0.99, r / max(dist, 1e-3)))) * 1.1
            V = gl_look_at(eye, c)
            P = gl_perspective(min(fov, 150.0), 1.0, max(0.01, dist - r * 2), dist + r * 2)
        return P @ V

    # ---------------------------------------------------------------- render
    def render(self, scene: RenderScene, K: np.ndarray, width: int, height: int, T_world_cam: np.ndarray,
               near: float = 0.01, far: float = 20.0, solo_ids: Optional[List[int]] = None,
               read_color: bool = True, read_normals: bool = True) -> RenderOutput:
        ctx = self.ctx
        fb = self._fbo(width, height)
        P = gl_projection_from_K(K, width, height, near, far)
        V = gl_view_from_cv(T_world_cam)
        cam_pos = T_world_cam[:3, 3]
        light_vp = self._shadow_matrix(scene)
        drawables = [(inst, self.get_mesh(inst.asset), self._model_matrix(inst)) for inst in scene.instances]
        ground = scene.ground

        # ------------------------------------------------ shadow pass
        if light_vp is not None:
            self.shadow_fbo.use()
            self.shadow_fbo.clear(depth=1.0)
            ctx.viewport = (0, 0, self.shadow_size, self.shadow_size)
            self._mat(self.prog_sh, "V", np.eye(4))
            self._mat(self.prog_sh, "P", light_vp)
            for inst, parts, M in drawables:
                self._mat(self.prog_sh, "M", M)
                for p in parts:
                    p.vao_sh.render()

        # ------------------------------------------------ colour pass
        prog = self.prog_pbr
        target = fb["ms_fbo"] if self.msaa > 1 else fb["col_fbo"]
        target.use()
        ctx.viewport = (0, 0, width, height)
        target.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self._mat(prog, "V", V)
        self._mat(prog, "P", P)
        self._mat(prog, "light_vp", light_vp if light_vp is not None else np.eye(4))
        self._w(prog, "cam_pos", tuple(float(x) for x in cam_pos))
        lp = np.zeros((MAX_LIGHTS, 4), np.float32)
        lc = np.zeros((MAX_LIGHTS, 3), np.float32)
        for i, L in enumerate(scene.lights[:MAX_LIGHTS]):
            v = np.asarray(L.vec, np.float64)
            if L.kind == "directional":
                v = v / (np.linalg.norm(v) + 1e-12)
            lp[i, :3], lp[i, 3] = v, 0.0 if L.kind == "directional" else 1.0
            lc[i] = L.color
        if "light_pos" in prog:
            prog["light_pos"].write(lp.tobytes())
            prog["light_col"].write(lc.tobytes())
        self._w(prog, "n_lights", min(len(scene.lights), MAX_LIGHTS))
        self._w(prog, "amb_sky", tuple(float(x) for x in scene.ambient_sky))
        self._w(prog, "amb_ground", tuple(float(x) for x in scene.ambient_ground))
        self._w(prog, "shadow_on", 1 if light_vp is not None else 0)
        self._w(prog, "shadow_texel", 1.0 / self.shadow_size)
        self._w(prog, "shadow_bias", 0.0008)
        self._w(prog, "shadow_radius", float(scene.shadow_softness))
        self._w(prog, "catcher", 0)
        self.shadow_tex.use(location=1)
        self._w(prog, "shadow_map", 1)
        self._w(prog, "tex", 0)

        def draw_pbr(part, M, base, tint, rough, metal, use_tex, uv_mode=0, uv_rot=None, uv_scale=(1.0, 1.0), tex=None):
            self._mat(prog, "M", M)
            self._w(prog, "base_color", tuple(float(x) for x in base))
            self._w(prog, "tint", tuple(float(x) for x in tint))
            self._w(prog, "roughness", float(rough))
            self._w(prog, "metallic", float(metal))
            t = tex if tex is not None else (part.texture if use_tex else None)
            self._w(prog, "use_texture", 1 if t is not None else 0)
            (t or self.white).use(location=0)
            self._w(prog, "use_vcolor", 1 if (part.has_color and use_tex) else 0)
            self._w(prog, "uv_mode", uv_mode)
            if uv_rot is not None and "uv_rot" in prog:
                prog["uv_rot"].write(np.asarray(uv_rot, np.float32).T.tobytes())
            self._w(prog, "uv_scale", (float(uv_scale[0]), float(uv_scale[1])))
            part.vao_pbr.render()

        for inst, parts, M in drawables:
            for p in parts:
                if inst.base_color is not None:
                    base = (*inst.base_color, 1.0)
                    use_tex = False
                else:
                    base = (*_srgb_to_lin(p.base_color[:3]), 1.0)
                    use_tex = inst.use_texture
                draw_pbr(p, M, base, inst.tint, inst.roughness, inst.metallic, use_tex)

        if ground.enabled and ground.visible:
            gp = self._ground(ground.extent)
            a = math.radians(ground.rotation_deg)
            rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            gt = self._ground_texture(ground)
            if ground.catcher:
                self._w(prog, "catcher", 1)
                self._w(prog, "catcher_strength", float(ground.catcher_strength))
            draw_pbr(gp, np.eye(4), (*ground.color, 1.0), (1, 1, 1), ground.roughness, 0.0, gt is not None,
                     uv_mode=1, uv_rot=rot, tex=gt,
                     uv_scale=(max(ground.tile_size, 1e-3), max(ground.tile_size_y or ground.tile_size, 1e-3)))

        if self.msaa > 1:
            ctx.copy_framebuffer(fb["col_fbo"], fb["ms_fbo"])
        self.last_color, self.last_size = fb["col"], (width, height)
        rgba = None
        if read_color:
            rgba = np.frombuffer(fb["col_fbo"].read(components=4, dtype="f4", attachment=0), np.float32)
            rgba = rgba.reshape(height, width, 4)[::-1].copy()

        # ------------------------------------------------ gbuffer pass
        gprog = self.prog_gb
        fb["gb_fbo"].use()
        ctx.viewport = (0, 0, width, height)
        fb["gb_fbo"].clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self._mat(gprog, "V", V)
        self._mat(gprog, "P", P)
        self._mat(gprog, "light_vp", np.eye(4))
        if "V3" in gprog:
            gprog["V3"].write(np.asarray(V[:3, :3], np.float32).T.tobytes())
        for inst, parts, M in drawables:
            self._mat(gprog, "M", M)
            self._w(gprog, "inst_id", float(inst.inst_id))
            for p in parts:
                p.vao_gb.render()
        if ground.enabled and ground.visible:
            self._mat(gprog, "M", np.eye(4))
            self._w(gprog, "inst_id", 0.0)
            self._ground(ground.extent).vao_gb.render()
        inst_img = np.frombuffer(fb["gb_fbo"].read(components=1, dtype="f4", attachment=0), np.float32)
        inst_img = np.rint(inst_img.reshape(height, width)[::-1]).astype(np.int32)
        depth = np.frombuffer(fb["gb_fbo"].read(components=1, dtype="f4", attachment=1), np.float32)
        depth = depth.reshape(height, width)[::-1].copy()
        nrm = None
        if read_normals:
            nrm = np.frombuffer(fb["gb_fbo"].read(components=4, dtype="f4", attachment=2), np.float32)
            nrm = nrm.reshape(height, width, 4)[::-1, :, :3].copy()

        # ------------------------------------------------ solo (amodal) passes
        solo: Dict[int, Tuple[np.ndarray, Tuple[int, int]]] = {}
        if solo_ids:
            sprog = self.prog_solo
            fb["solo_fbo"].use()
            ctx.viewport = (0, 0, width, height)
            self._mat(sprog, "V", V)
            self._mat(sprog, "P", P)
            self._mat(sprog, "light_vp", np.eye(4))
            T_cam_world = inv_T(T_world_cam)
            want = set(solo_ids)
            for inst, parts, M in drawables:
                if inst.inst_id not in want:
                    continue
                rect = self._screen_rect(inst, M, T_cam_world, K, width, height, near)
                if rect is None:
                    solo[inst.inst_id] = (np.zeros((0, 0), bool), (0, 0))
                    continue
                x0, y0, x1, y1 = rect
                fb["solo_fbo"].clear(0.0, depth=1.0)
                self._mat(sprog, "M", M)
                for p in parts:
                    p.vao_solo.render()
                # GL origin bottom-left
                vp = (x0, height - y1, x1 - x0, y1 - y0)
                buf = fb["solo_fbo"].read(viewport=vp, components=1, dtype="f1", attachment=0)
                m = np.frombuffer(buf, np.uint8).reshape(y1 - y0, x1 - x0)[::-1] > 127
                solo[inst.inst_id] = (m.copy(), (x0, y0))
        return RenderOutput(rgba=rgba, inst=inst_img, depth=depth, normals=nrm, solo=solo)

    # ---------------------------------------------------------------- GPU sensor model
    def _pp_fbo(self, w: int, h: int) -> dict:
        key = (w, h)
        if key not in self._pp_fbos:
            if len(self._pp_fbos) > 4:
                for o in self._pp_fbos.pop(next(iter(self._pp_fbos))).values():
                    o.release()
            d = {}
            for n in ("a", "b"):
                t = self.ctx.texture((w, h), 4, dtype="f2")
                t.filter = (moderngl.LINEAR, moderngl.LINEAR)
                t.repeat_x = t.repeat_y = False
                d[n] = t
                d[n + "_fbo"] = self.ctx.framebuffer([t])
            d["out"] = self.ctx.renderbuffer((w, h), 4)
            d["out_fbo"] = self.ctx.framebuffer([d["out"]])
            d["bg"] = self.ctx.texture((w, h), 3)
            self._pp_fbos[key] = d
        return self._pp_fbos[key]

    def _map_texture(self, canvas) -> moderngl.Texture:
        key = id(canvas.map[0])
        if key not in self._map_tex:
            if len(self._map_tex) > 8:
                self._map_tex.pop(next(iter(self._map_tex)))[1].release()
            m = np.dstack([canvas.map[0], canvas.map[1]])[::-1]
            t = self.ctx.texture((canvas.W, canvas.H), 2, np.ascontiguousarray(m, np.float32).tobytes(), dtype="f4")
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._map_tex[key] = (canvas.map, t)  # keep the arrays alive so id() stays unique
        return self._map_tex[key][1]

    def _pass(self, name: str, target, **uniforms):
        prog = self.pp[name]
        target.use()
        self.ctx.viewport = (0, 0, *target.size)
        for k, v in uniforms.items():
            if k in prog:
                prog[k].value = v
        self.pp_vao[name].render(moderngl.TRIANGLES, vertices=3)

    def postprocess(self, canvas, bg_u8: Optional[np.ndarray], bg_gain: float, p: dict,
                    keep_alpha: bool = False, bg_is_photo: bool = False) -> np.ndarray:
        """GPU sensor model applied to the last rendered colour buffer: crop / lens distortion, background
        composite, exposure, white balance, vignetting, defocus, motion blur, shot + read noise, tone mapping,
        saturation / contrast. Returns uint8 HxWx3 (HxWx4 with keep_alpha). Same model as apply_sensor_cpu."""
        assert self.last_color is not None, "render() first"
        W, H = canvas.W, canvas.H
        f = self._pp_fbo(W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        try:
            self.last_color.use(location=0)
            if canvas.distorted:
                self._map_texture(canvas).use(location=1)
            if bg_u8 is not None:
                f["bg"].write(np.ascontiguousarray(bg_u8[::-1]).tobytes())
                f["bg"].use(location=2)
            en = p.get("enabled", True)
            post_ = bool(bg_is_photo and bg_u8 is not None)
            ae = en and p.get("ae", False) and not post_
            gain = (np.asarray(p["wb"]) * (1.0 if ae else p["exposure"])) if en else np.ones(3)
            post = bool(bg_is_photo and bg_u8 is not None)
            self._pass("compose", f["a_fbo"], canvas=0, map=1, bg=2, distorted=int(canvas.distorted),
                       offset=(int(canvas.px), int(canvas.py)), canvas_size=(float(canvas.Wr), float(canvas.Hr)),
                       out_size=(float(W), float(H)), has_bg=int(bg_u8 is not None and not post),
                       bg_gain=float(bg_gain),
                       gain=tuple(float(g) for g in gain), vignette=float(p.get("vignetting", 0.0)) if en else 0.0)
            ae_gain = 1.0
            if ae:   # measure the composite on a small mip level, solve the gain on the CPU (cheap)
                from .imaging import auto_exposure_gain
                t = f["a"]
                t.build_mipmaps()
                lvl = max(0, int(np.ceil(np.log2(max(W, H) / 96.0))))
                lw, lh = max(1, W >> lvl), max(1, H >> lvl)
                small = np.frombuffer(t.read(level=lvl), np.float16).astype(np.float32).reshape(lh, lw, 4)[..., :3]
                t.filter = (moderngl.LINEAR, moderngl.LINEAR)
                ae_gain = auto_exposure_gain(small, float(p["ae_target"]), p.get("tonemap", "filmic"),
                                             float(p.get("contrast", 1.0)))
                p["ae_gain"] = ae_gain
            cur, other = "a", "b"
            sigma = float(p.get("defocus_sigma", 0.0)) if en else 0.0
            if sigma > 0.05:
                for d in ((1, 0), (0, 1)):
                    f[cur].use(location=0)
                    self._pass("gauss", f[other + "_fbo"], src=0, dir=d, sigma=sigma)
                    cur, other = other, cur
            mb = p.get("motion_blur") if en else None
            if mb:
                L, ang = int(mb[0]), np.radians(mb[1])
                f[cur].use(location=0)
                self._pass("motion", f[other + "_fbo"], src=0, dir=(float(np.cos(ang)), float(np.sin(ang))), L=L)
                cur, other = other, cur
            f[cur].use(location=0)
            if post:
                f["bg"].use(location=2)
            self._pass("final", f["out_fbo"], src=0, bg=2, post_bg=int(post), bg_gain=float(bg_gain),
                       ae_gain=float(ae_gain),
                       shot=float(p.get("shot_noise", 0.0)) if en else 0.0,
                       readn=float(p.get("gaussian_noise", 0.0)) if en else 0.0,
                       seed=int(p.get("noise_seed", 0)) & 0x7FFFFFFF,
                       filmic=int(p.get("tonemap", "srgb") == "filmic"),
                       sat=float(p.get("saturation", 1.0)) if en else 1.0,
                       con=float(p.get("contrast", 1.0)) if en else 1.0, keep_alpha=int(keep_alpha))
            img = np.frombuffer(f["out_fbo"].read(components=4), np.uint8).reshape(H, W, 4)[::-1]
        finally:
            self.ctx.enable(moderngl.DEPTH_TEST)
        return np.ascontiguousarray(img if keep_alpha else img[..., :3])

    @staticmethod
    def _screen_rect(inst: Instance, M: np.ndarray, T_cam_world: np.ndarray, K: np.ndarray, w: int, h: int,
                     near: float):
        corners = inst.asset.bbox_corners()
        Pc = transform_points(T_cam_world @ M, corners)
        if np.all(Pc[:, 2] <= near):
            return None
        if np.any(Pc[:, 2] <= near):
            return 0, 0, w, h
        uv = Pc[:, :2] / Pc[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
        x0 = int(max(0, math.floor(uv[:, 0].min()) - 2))
        y0 = int(max(0, math.floor(uv[:, 1].min()) - 2))
        x1 = int(min(w, math.ceil(uv[:, 0].max()) + 3))
        y1 = int(min(h, math.ceil(uv[:, 1].max()) + 3))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
