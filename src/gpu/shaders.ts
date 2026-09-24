// Adapted from https://newton-fractal.pages.dev/ (ship material/perturbation/rings).
// FE operation order, branch boundaries, escape radius and smooth metric are preserved.
// No fast-math, reduced precision, reduced iteration budgets or lower AA counts.
export const vertexShader = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

const shipSource = `precision highp float;
precision highp int;

in vec2 vUv;
out vec4 fragmentColour;
uniform vec2 center;
uniform float scale;
#ifdef SHIP_FULL
#define fold 1.0
#define celtic 0.0
#else
uniform float fold;
uniform float celtic;
#endif
uniform float hue;
uniform float aspect;
uniform vec2 res;
uniform float aa;
uniform vec2 jitter;
uniform int iterations;

#define SHIP_LOOP_LIMIT iterations

float shipIterationMetric = 0.0;
#if !defined(SHIP_PROBE) && !defined(SHIP_BATCH_STEPS)

#endif

vec3 shipColour(float smoothIteration) {
    shipIterationMetric = smoothIteration;
#ifdef SHIP_PROBE
    float value = clamp(floor(smoothIteration * 16.0), 1.0, 16777215.0);
    return vec3(mod(value, 256.0), mod(floor(value / 256.0), 256.0), floor(value / 65536.0)) / 255.0;
#else
    vec3 colour = 0.5 + 0.5 * cos(0.18 * smoothIteration + hue + vec3(0.0, 0.7, 1.5));
    return colour * (0.3 + 0.7 * (1.0 - exp(-0.08 * smoothIteration)));
#endif
}

vec3 shipInterior() {
    shipIterationMetric = 0.0;
#ifdef SHIP_PROBE
    return vec3(0.0);
#else
    return vec3(0.006, 0.008, 0.012);
#endif
}

#ifdef SHIP_RING
// Ring maps (rings.ts): x is the angle, y a ring of the current block, from its
// outermost ring inwards. viewScale holds the outermost ring's radius.
uniform vec4 ringBlock; // first x, first row, angles, log2 radius step
vec2 shipOffset(vec2 uv) {
    vec2 texel = floor(gl_FragCoord.xy) - ringBlock.xy;
    float angle = 6.283185307179586 * texel.x / ringBlock.z;
    return exp2(-texel.y * ringBlock.w) * vec2(cos(angle), sin(angle));
}
#else
// Offset from the view centre in units of the view height.
vec2 shipOffset(vec2 uv) { return vec2((uv.x - 0.5) * aspect, 0.5 - uv.y); }
#endif

vec3 sampleShipDirect(vec2 uv) {
    vec2 c = center + shipOffset(uv) * scale;
    vec2 z = vec2(0.0);
    int iteration = 0;
#ifdef SHIP_BATCH_STEPS
    z = savedDelta.xy;
    iteration = savedIndex.y;
#endif
    for (int step = 0; step < SHIP_LOOP_LIMIT; step++) {
        if (iteration >= iterations) break;
        int i = iteration++;
        float xy = z.x * z.y;
        float real = z.x * z.x - z.y * z.y;
        z = vec2(real < 0.0 ? (1.0 - 2.0 * celtic) * real : real,
                 2.0 * (xy < 0.0 ? (1.0 - 2.0 * fold) * xy : xy)) + c;
        float radius2 = dot(z, z);
        if (radius2 > 65536.0) {
            float smoothIteration = float(i) + 2.0 - log2(log2(radius2) * 0.5);
            return shipColour(smoothIteration);
        }
    }
#ifdef SHIP_BATCH_STEPS
    if (iteration < iterations) suspendShip(vec4(z, 0.0, 0.0), 0, iteration);
#endif
    return shipInterior();
}

// Extended-range numbers: normalized mantissa and a separate binary exponent.
// Deltas retain their range even when the view is smaller than 1e-1000.
// Sampler precision controls Float32 reads independently of highp float.
// Mobile lowp reads can round the orbit or lose the BLA matrices/error bounds.
uniform highp sampler2D referenceOrbit;
uniform highp sampler2D realOrbit;
uniform highp sampler2D blaA;
uniform highp sampler2D blaB;
uniform highp sampler2D blaBounds;
uniform vec2 referenceSize;
uniform vec2 offsetX;
uniform vec2 offsetY;
uniform vec2 viewScale;
uniform int referenceLength;

vec2 normalizeFE(float m, float e) {
    if (m == 0.0) return vec2(0.0);
    float shift = floor(log2(abs(m)));
    return vec2(m * exp2(-shift), e + shift);
}
vec2 numberFE(float x) { return normalizeFE(x, 0.0); }
vec2 addFE(vec2 a, vec2 b) {
    if (a.x == 0.0) return b;
    if (b.x == 0.0) return a;
    float difference = a.y - b.y;
    if (difference > 30.0) return a;
    if (difference < -30.0) return b;
    float e = max(a.y, b.y);
    return normalizeFE(a.x * exp2(a.y - e) + b.x * exp2(b.y - e), e);
}
vec2 negateFE(vec2 a) { return vec2(-a.x, a.y); }
vec2 absFE(vec2 a) { return vec2(abs(a.x), a.y); }
bool lessFE(vec2 a, vec2 b) {
    if (a.x == 0.0) return b.x != 0.0;
    if (b.x == 0.0) return false;
    return a.y < b.y || (a.y == b.y && abs(a.x) < abs(b.x));
}
vec2 maxFE(vec2 a, vec2 b) { return lessFE(a, b) ? absFE(b) : absFE(a); }
vec2 multiplyFE(vec2 a, vec2 b) {
    // A zero mantissa stays zero; its exponent is immaterial.
    float m = a.x * b.x;
    return abs(m) >= 2.0 ? vec2(m * 0.5, a.y + b.y + 1.0) : vec2(m, a.y + b.y);
}
vec2 timesFE(vec2 a, float b) { return multiplyFE(a, numberFE(b)); }
vec2 twiceFE(vec2 a) { return vec2(a.x, a.y + 1.0); }
float valueFE(vec2 a) { return a.x * exp2(clamp(a.y, -126.0, 126.0)); }

// |reference + delta| - |reference|, without subtracting near-equal floats.
vec2 diffAbsFE(vec2 reference, vec2 delta) {
    if (reference.x == 0.0) return absFE(delta);
    if (reference.x * delta.x >= 0.0 || lessFE(delta, reference))
        return reference.x > 0.0 ? delta : negateFE(delta);
    vec2 sum = addFE(twiceFE(reference), delta);
    return delta.x > 0.0 ? sum : negateFE(sum);
}
// texelFetch reads these texels without sampling. With texture2D, ANGLE's
// D3D11 compiler derived gradients inside the orbit loops: deep views ran up
// to three times slower.
vec4 orbitAt(int index) {
    return texelFetch(referenceOrbit, ivec2(index % SHIP_REFERENCE_WIDTH, index / SHIP_REFERENCE_WIDTH), 0);
}
vec2 realAt(int index) {
    return texelFetch(realOrbit, ivec2(index % SHIP_REFERENCE_WIDTH, index / SHIP_REFERENCE_WIDTH), 0).xy;
}
// The aligned block of 2^level steps starting at index. All levels share the
// orbit's texture size; see reference.ts for the layout.
ivec2 blaAt(int index, int level) {
    int capacity = int(referenceSize.x * referenceSize.y);
    int block = capacity - (capacity >> (level - 1)) + (index >> level);
    return ivec2(block % SHIP_REFERENCE_WIDTH, block / SHIP_REFERENCE_WIDTH);
}
// The longest aligned block that may start at a positive index within the
// budget. The lowest set bit is the alignment; its logarithm is exact.
int blaTopLevel(int index, int iteration) {
    #ifdef SHIP_OPTIMIZED
    uint bits = uint(index);
    int level = 0;
    if ((bits & 65535u) == 0u) { level += 16; bits >>= 16; }
    if ((bits & 255u) == 0u) { level += 8; bits >>= 8; }
    if ((bits & 15u) == 0u) { level += 4; bits >>= 4; }
    if ((bits & 3u) == 0u) { level += 2; bits >>= 2; }
    if ((bits & 1u) == 0u) level++;
    level = min(SHIP_BLA_LEVELS, level);
#else
    int level = min(SHIP_BLA_LEVELS, int(log2(float(index & -index)) + 0.5));
#endif
    while (level > 0 && (index + (1 << level) >= referenceLength || iteration + (1 << level) > iterations))
        level--;
    return level;
}
float log2FE(vec2 a) { return a.y + log2(abs(a.x)); }
vec2 scaleFE(vec2 a, float exponent) { return vec2(a.x, a.y + exponent); }
vec2 matrixRowFE(vec2 x, vec2 y, vec2 row) {
    return addFE(timesFE(x, row.x), timesFE(y, row.y));
}

float diffAbs(float c, float d) {
    return c >= 0.0 ? (c + d >= 0.0 ? d : -(2.0 * c + d)) : (c + d > 0.0 ? 2.0 * c + d : -d);
}

// Difference of (1-fold)*xy + fold*|xy|, preserving small orbit deltas.
// Keep the established full-Ship path at fold=1, including its rounding order.
// Apply each branch's slope directly. Blending two rounded products introduces
// errors even when that slope is exactly 0 or 0.5, amplified by deep iteration.
float foldedProductDifference(vec2 r, vec2 d) {
    if (fold == 1.0) {
        float ax = diffAbs(r.x, d.x), ay = diffAbs(r.y, d.y);
        return abs(r.x) * ay + abs(r.y) * ax + ax * ay;
    }
    float delta = r.x * d.y + r.y * d.x + d.x * d.y;
    float product = r.x * r.y;
    float slope = 1.0 - 2.0 * fold;
    if (product >= 0.0)
        return product + delta >= 0.0 ? delta : slope * delta - 2.0 * fold * product;
    return product + delta <= 0.0 ? slope * delta : delta + 2.0 * fold * product;
}

float foldedDifference(float reference, float delta, float strength) {
    float slope = 1.0 - 2.0 * strength;
    if (reference >= 0.0)
        return reference + delta >= 0.0 ? delta : slope * delta - 2.0 * strength * reference;
    return reference + delta <= 0.0 ? slope * delta : delta + 2.0 * strength * reference;
}
vec2 foldedDifferenceFE(vec2 reference, vec2 delta, float strength) {
    float slope = 1.0 - 2.0 * strength;
    if (reference.x == 0.0) return delta.x < 0.0 ? timesFE(delta, slope) : delta;
    if (reference.x * delta.x >= 0.0 || lessFE(delta, reference))
        return reference.x > 0.0 ? delta : timesFE(delta, slope);
    return reference.x > 0.0
        ? addFE(timesFE(delta, slope), timesFE(reference, -2.0 * strength))
        : addFE(delta, timesFE(reference, 2.0 * strength));
}
float realDifference(vec2 r, vec2 d, int index) {
    float delta = 2.0 * (r.x * d.x - r.y * d.y) + d.x * d.x - d.y * d.y;
    if (celtic == 0.0) return delta;
    vec2 q = realAt(index);
    return foldedDifference(q.x * exp2(q.y), delta, celtic);
}
// Once deltas grow into the normal float range, keep them there across steps.
// Products of inputs above 2^-60 stay normal. Retry a small result or a tiny
// reference with extended-range arithmetic BEFORE committing the step, so
// cancellation and rebasing cannot discard the original sub-float dc.
bool continueShipFloat(inout vec2 d, inout int m, inout int n, vec2 dc, out vec3 colour) {
    // Each step's next reference is the following step's reference; a rebase
    // restarts at the zero orbit point. Carry it instead of reading it twice.
    vec4 rfe = orbitAt(m);
    vec2 r = vec2(valueFE(rfe.xy), valueFE(rfe.zw));
    for (int step = 0; step < SHIP_LOOP_LIMIT; step++) {
        if (n >= iterations) {
            colour = shipInterior();
            return true;
        }
#ifdef SHIP_BATCH_STEPS
        if (work >= SHIP_BATCH_STEPS) break;
#endif
        #ifndef SHIP_OPTIMIZED
        rfe = orbitAt(m);
        r = vec2(valueFE(rfe.xy), valueFE(rfe.zw));
#endif
        if ((rfe.x != 0.0 && rfe.y < -60.0) || (rfe.z != 0.0 && rfe.w < -60.0)) break;
        if (celtic > 0.0) {
            vec2 q = realAt(m);
            if (q.x != 0.0 && q.y < -120.0) break;
        }
        vec2 nextD = vec2(realDifference(r, d, m),
                     2.0 * foldedProductDifference(r, d)) + dc;
        if (min(abs(nextD.x), abs(nextD.y)) <= 1e-15) break;
        vec4 nextR = orbitAt(m + 1);
        if ((nextR.x != 0.0 && nextR.y < -60.0) || (nextR.z != 0.0 && nextR.w < -60.0)) break;
        vec2 nextRValue = vec2(valueFE(nextR.xy), valueFE(nextR.zw));
        vec2 z = nextRValue + nextD;
        float radius2 = dot(z, z);
        if (radius2 > 65536.0) {
            colour = shipColour(float(n + 1) + 1.0 - log2(log2(radius2) * 0.5));
            return true;
        }
        bool rebase = m + 1 >= referenceLength - 1 || max(abs(z.x),abs(z.y)) < max(abs(nextD.x),abs(nextD.y));
        if (rebase && min(abs(z.x), abs(z.y)) <= 1e-15) break;
        d = rebase ? z : nextD;
        m = rebase ? 0 : m + 1;
        rfe = rebase ? vec4(0.0) : nextR;
        r = rebase ? vec2(0.0) : nextRValue;
        n++;
#ifdef SHIP_BATCH_STEPS
        work++;
#endif
    }
    return false;
}

vec3 sampleShipPerturbed(vec2 uv) {
    vec2 p = shipOffset(uv);
    vec2 dcx = addFE(offsetX, timesFE(viewScale, p.x));
    vec2 dcy = addFE(offsetY, timesFE(viewScale, p.y));
    float dcLog = log2FE(maxFE(dcx, dcy));
    vec2 dx = vec2(0.0), dy = vec2(0.0);
    int referenceIndex = 0;
    int iteration = 0;
#ifdef SHIP_BATCH_STEPS
    dx = savedDelta.xy;
    dy = savedDelta.zw;
    referenceIndex = savedIndex.x;
    iteration = savedIndex.y;
#endif
    // The reference at referenceIndex, carried between steps.
    vec4 reference = orbitAt(referenceIndex);
    for (int step = 0; step < SHIP_LOOP_LIMIT; step++) {
        if (iteration >= iterations) break;
#ifdef SHIP_BATCH_STEPS
        if (work >= SHIP_BATCH_STEPS) break;
#endif
        if (min(dx.y, dy.y) > -50.0 && max(dx.y, dy.y) > -40.0 && dx.x != 0.0 && dy.x != 0.0) {
            vec2 d = vec2(valueFE(dx), valueFE(dy));
            // Underflowed dc is negligible for accepted native results, but
            // keep dcx/dcy intact for the extended-range fallback below.
            vec2 dc = vec2(dcx.y < -126.0 ? 0.0 : valueFE(dcx), dcy.y < -126.0 ? 0.0 : valueFE(dcy));
            vec3 colour;
            if (continueShipFloat(d, referenceIndex, iteration, dc, colour)) return colour;
            dx = numberFE(d.x);
            dy = numberFE(d.y);
            reference = orbitAt(referenceIndex);
#ifdef SHIP_BATCH_STEPS
            if (iteration >= iterations || work >= SHIP_BATCH_STEPS) break;
#endif
        }
        bool skipped = false;
        // Try the longest aligned block first; each is valid only if its first
        // half is. Matrices are scaled by 2^bounds.zw, radii stored as log2.
        int top = referenceIndex > 0 ? blaTopLevel(referenceIndex, iteration) : 0;
        float deltaLog = top > 0 ? log2FE(maxFE(dx, dy)) : 0.0;
        for (int level = top; level > 0; level--) {
            ivec2 at = blaAt(referenceIndex, level);
            vec4 bounds = texelFetch(blaBounds, at, 0);
            if (deltaLog < bounds.x && dcLog < bounds.y) {
                vec4 A = texelFetch(blaA, at, 0), B = texelFetch(blaB, at, 0);
                vec2 nextX = addFE(scaleFE(matrixRowFE(dx, dy, A.xy), bounds.z), scaleFE(matrixRowFE(dcx, dcy, B.xy), bounds.w));
                dy = addFE(scaleFE(matrixRowFE(dx, dy, A.zw), bounds.z), scaleFE(matrixRowFE(dcx, dcy, B.zw), bounds.w));
                dx = nextX;
                iteration += 1 << level;
                referenceIndex += 1 << level;
                skipped = true;
                break;
            }
        }
        if (!skipped) {
            #ifndef SHIP_OPTIMIZED
            reference = orbitAt(referenceIndex);
#endif
            vec2 X = reference.xy, Y = reference.zw;
            vec2 realDelta = addFE(twiceFE(addFE(multiplyFE(X, dx), negateFE(multiplyFE(Y, dy)))),
                                  addFE(multiplyFE(dx, dx), negateFE(multiplyFE(dy, dy))));
            if (celtic > 0.0) realDelta = foldedDifferenceFE(realAt(referenceIndex), realDelta, celtic);
            vec2 nextX = addFE(realDelta, dcx);
            vec2 productDelta;
            if (fold == 1.0) {
                vec2 ax = diffAbsFE(X, dx), ay = diffAbsFE(Y, dy);
                productDelta = addFE(addFE(multiplyFE(absFE(X), ay), multiplyFE(absFE(Y), ax)), multiplyFE(ax, ay));
            } else {
                productDelta = addFE(addFE(multiplyFE(X, dy), multiplyFE(Y, dx)), multiplyFE(dx, dy));
                if (fold > 0.0)
                    productDelta = foldedDifferenceFE(multiplyFE(X, Y), productDelta, fold);
            }
            dy = addFE(twiceFE(productDelta), dcy);
            dx = nextX;
            iteration++;
            referenceIndex++;
        }
        reference = orbitAt(referenceIndex);
        vec2 totalX = addFE(reference.xy, dx), totalY = addFE(reference.zw, dy);
        vec2 totalSize = maxFE(totalX, totalY);
        if (!lessFE(totalSize, vec2(1.0, 8.0))) {
            float x = valueFE(totalX), y = valueFE(totalY);
            float radius2 = min(x * x + y * y, 1e30);
            float smoothIteration = float(iteration) + 1.0 - log2(log2(radius2) * 0.5);
            return shipColour(smoothIteration);
        }
        // Rebase after cancellation or before exhausting an escaping reference.
        if (referenceIndex >= referenceLength - 1 || lessFE(totalSize, maxFE(dx, dy))) {
            dx = totalX;
            dy = totalY;
            referenceIndex = 0;
            reference = vec4(0.0);
        }
#ifdef SHIP_BATCH_STEPS
        work++;
#endif
    }
#ifdef SHIP_BATCH_STEPS
    if (iteration < iterations)
        suspendShip(vec4(dx, dy), referenceIndex, iteration);
#endif
    return shipInterior();
}

vec2 orbitFloat(int index) {
    vec4 r = orbitAt(index);
    return vec2(r.x * exp2(r.y), r.z * exp2(r.w));
}
vec3 sampleShipFloat(vec2 uv) {
    vec2 dc = vec2(valueFE(offsetX), valueFE(offsetY)) +
        shipOffset(uv) * valueFE(viewScale);
    float dcLog = log2(max(abs(dc.x), abs(dc.y)));
    vec2 d = vec2(0.0);
    int m = 0, n = 0;
#ifdef SHIP_BATCH_STEPS
    d = savedDelta.xy;
    m = savedIndex.x;
    n = savedIndex.y;
#endif
    // The reference at m, carried from the previous step's escape check.
    vec2 r = orbitFloat(m);
    for (int step = 0; step < SHIP_LOOP_LIMIT; step++) {
        if (n >= iterations) break;
        bool skipped = false;
        // Try the longest aligned block first, as in the extended-range loop.
        // Blocks under 8 steps rarely hold at float scales; skip their lookups.
        int top = m > 0 && (m & 7) == 0 ? blaTopLevel(m, n) : 0;
        float deltaLog = top >= 3 ? log2(max(abs(d.x), abs(d.y))) : 0.0;
        for (int level = top; level >= 3; level--) {
            ivec2 at = blaAt(m, level);
            vec4 bounds = texelFetch(blaBounds, at, 0);
            // Where these bounds hold at float scales, 2^bounds.zw stays finite.
            if (deltaLog < bounds.x && dcLog < bounds.y && max(bounds.z, bounds.w) < 126.0) {
                vec4 A = texelFetch(blaA, at, 0), B = texelFetch(blaB, at, 0);
                d = vec2(dot(A.xy, d), dot(A.zw, d)) * exp2(bounds.z) +
                    vec2(dot(B.xy, dc), dot(B.zw, dc)) * exp2(bounds.w);
                n += 1 << level;
                m += 1 << level;
                skipped = true;
                break;
            }
        }
        if (!skipped) {
            #ifndef SHIP_OPTIMIZED
            r = orbitFloat(m);
#endif
            d = vec2(realDifference(r, d, m),
                     2.0 * foldedProductDifference(r, d)) + dc;
            n++; m++;
        }
        r = orbitFloat(m);
        vec2 z = r + d;
        float radius2 = dot(z, z);
        if (radius2 > 65536.0) return shipColour(float(n) + 1.0 - log2(log2(radius2) * 0.5));
        if (m >= referenceLength - 1 || radius2 < dot(d, d)) { d = z; m = 0; r = vec2(0.0); }
    }
#ifdef SHIP_BATCH_STEPS
    if (n < iterations) suspendShip(vec4(d, 0.0, 0.0), m, n);
#endif
    return shipInterior();
}

// Each program links one path; material.ts chooses it for the current view.
vec3 sampleShip(vec2 uv) {
#if SHIP_PATH == 0
    return sampleShipDirect(uv);
#elif SHIP_PATH == 1
    return sampleShipFloat(uv);
#else
    return sampleShipPerturbed(uv);
#endif
}


void main() {
#ifdef SHIP_BATCH_STEPS
    savedDelta = vec4(0.0);
    savedIndex = ivec2(0);
    if (firstBatch < 0.5) {
        savedDelta = texture(previousDelta, gl_FragCoord.xy / stateSize);
        vec4 status = texture(previousStatus, gl_FragCoord.xy / stateSize);
        // Completed pixels carry their colour through the remaining passes.
        if (status.z > 0.5) {
            nextDelta = savedDelta;
            nextStatus = status;
            return;
        }
        savedIndex = ivec2(status.xy);
    }
    vec3 colour = sampleShip(vUv + jitter / res);
    if (!suspended) {
        nextDelta = vec4(colour, shipIterationMetric);
        nextStatus = vec4(0.0, 0.0, 1.0, 0.0);
    }
#elif defined(SHIP_PROBE) || defined(SHIP_RING)
    fragmentColour = vec4(sampleShip(vUv), 1.0);
#else
    vec3 colour = vec3(0.0);
    float metric = 0.0;
    for (int i = 0; i < 5; i++) {
        if (float(i) >= aa) break;
        vec2 offset = aa < 1.5 ? vec2(0.0) :
            vec2((float(i) + 0.5) / aa - 0.5, fract((float(i) + 0.5) * 0.61803398875) - 0.5);
        colour += sampleShip(vUv + (offset + jitter) / res);
        metric += shipIterationMetric;
    }
    fragmentColour = vec4(colour / aa, 1.0);
#endif
}
`;

// Constant fold/Celtic specialization was rejected by browser pixel tests:
// compiling them as constants changes GPU rounding/fusion, even on the direct
// path. Keep uniforms for both variants; only alignment/fetch changes remain.
export function shipFragment(path: 0 | 1 | 2, optimized: boolean, _fullShip: boolean, ring = false): string {
    return `#version 300 es
#define SHIP_BLA_LEVELS 10
#define SHIP_REFERENCE_WIDTH 1024
#define SHIP_PATH ${path}
${optimized ? '#define SHIP_OPTIMIZED 1' : ''}
${ring ? '#define SHIP_RING 1' : ''}
${shipSource}`;
}

export const ringFragment = `#version 300 es
#define SHIP_RING_BANDS 16
precision highp float;
precision highp int;
precision highp int;

// Assembles a frame from the ring map built in rings.ts. Band j covers pixel
// radii (outer/2^(j+1), outer/2^j] with angles proportional to its radius, so
// every pixel spans one to two texels per axis. A 2x2 box of bilinear taps
// integrates that footprint.
in vec2 vUv;
out vec4 fragmentColour;
uniform highp sampler2D ringMap;
uniform float aspect;
uniform float viewHeight; // display pixels per view height
uniform int bandCount;
uniform vec4 bandLayout[SHIP_RING_BANDS]; // first row, angles, log2 step, rings
uniform vec3 bandPlace[SHIP_RING_BANDS]; // ring index at one view height (mod rings), outer radius, first column

vec3 ringTexel(ivec2 origin, ivec2 texel) {
    return texelFetch(ringMap, origin + texel, 0).rgb;
}

vec3 ringSample(vec4 band, float column, vec2 at) {
    vec2 base = floor(at), f = at - base;
    // Wrap angles and rings with integers: for an exact multiple, float mod can
    // return the divisor itself and read the next band's first ring.
    ivec2 size = ivec2(band.yw);
    ivec2 a = ivec2(mod(base, band.yw)) % size, b = (a + 1) % size;
    ivec2 origin = ivec2(int(column), int(band.x));
    return mix(
        mix(ringTexel(origin, a), ringTexel(origin, ivec2(b.x, a.y)), f.x),
        mix(ringTexel(origin, ivec2(a.x, b.y)), ringTexel(origin, b), f.x),
        f.y);
}

// Log2 width on each side of a band boundary where neighbours cross-fade
// (BLEND in rings.ts). Half an octave blends like trilinear mipmapping: only a
// band's centre is unmixed, so no radius switches between sample lattices.
const float BLEND = 0.5;

vec3 bandColour(int index, vec2 p, float radius) {
    vec4 band = bandLayout[index];
    float angles = band.y / 6.283185307179586;
    // Texel centres sit on the sampled angles and rings.
    vec2 at = vec2(atan(p.y, p.x) * angles,
                   bandPlace[index].x - log2(radius / viewHeight) / band.z);
    float quarter = angles / radius * 0.25;
    float column = bandPlace[index].z;
    return 0.25 * (ringSample(band, column, at + vec2(-quarter, -quarter)) +
                   ringSample(band, column, at + vec2(quarter, -quarter)) +
                   ringSample(band, column, at + vec2(-quarter, quarter)) +
                   ringSample(band, column, at + vec2(quarter, quarter)));
}

vec3 assembleRing(vec2 uv) {
    vec2 p = vec2((uv.x - 0.5) * aspect, 0.5 - uv.y);
    float innermost = bandPlace[bandCount - 1].y * 0.5;
    float radius = max(length(p) * viewHeight, innermost);
    float u = log2(bandPlace[0].y / radius);
    int index = clamp(int(floor(u)), 0, bandCount - 1);
    float d = u - float(index);
    vec3 colour = bandColour(index, p, radius);
    if (d > 1.0 - BLEND && index + 1 < bandCount)
        colour = mix(colour, bandColour(index + 1, p, radius), (d - 1.0 + BLEND) / (2.0 * BLEND));
    else if (d < BLEND && index > 0)
        colour = mix(bandColour(index - 1, p, radius), colour, (d + BLEND) / (2.0 * BLEND));
    return colour;
}

uniform vec2 res;
uniform float aa;
void main() {
    vec3 colour = vec3(0.0);
    for (int i = 0; i < 5; i++) {
        if (float(i) >= aa) break;
        vec2 offset = aa < 1.5 ? vec2(0.0) :
            vec2((float(i) + 0.5) / aa - 0.5, fract((float(i) + 0.5) * 0.61803398875) - 0.5);
        colour += assembleRing(vUv + offset / res);
    }
    fragmentColour = vec4(colour / aa, 1.0);
}
`;
