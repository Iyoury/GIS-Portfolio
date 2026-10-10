# Independent targets: the exact neutral spectrum, the Wright-Fisher matrix built in extended precision
# (mpmath binomials) with its eigenvalues from mpmath eig and the eigenvector by inverse iteration with a
# plain LU, and for N = 400 inverse iteration with a plain LU in mpmath at 400 digits (run once); no use
# of the Vandermonde structure.

# --- test case 0 ---
# one individual: the 2 x 2 matrix has the eigenvalues 1 and 1 - u - v with the right
# eigenvector (v, -u) (sign fixed by the normalisation); with u = 0 or v = 0 one state is absorbing and the
# mode vanishes there
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_call(N, s, u, v):
    out = fastest_mode(N, s, u, v)
    lam, mode = out
    assert isinstance(lam, float), lam
    mode = np.asarray(mode)
    assert mode.shape == (N + 1,) and np.issubdtype(mode.dtype, np.floating), (mode.shape, mode.dtype)
    return lam, mode

def _t_absorbing_mask(N, u, v):
    # mask of the absorbing states (i = 0 when v = 0, i = N when u = 0), where the exact mode is zero
    zero = np.zeros(N + 1, dtype=bool)
    zero[0] = not v > 0
    zero[N] = zero[N] or not u > 0
    return zero

def _t_check_mode(got, target, zero, tol, label):
    # entries whose exact value is zero (absorbing states) within 1e-250 of zero; every other entry within
    # tol times its magnitude (signs and the normalisation as stated)
    got = np.asarray(got, dtype=float)
    target = np.asarray(target, dtype=float)
    zero = np.asarray(zero, dtype=bool)
    assert np.all(np.abs(got[zero]) <= 1e-250), (label, got[zero])
    bad = ~zero & (np.abs(got - target) > tol * np.abs(target))
    assert not bad.any(), (label, np.nonzero(bad)[0][:5], got[bad][:5], target[bad][:5])

for _t_s, _t_u, _t_v in ((0.3, 0.05, 0.02), (-0.5, 1e-12, 0.1), (0.2, 0.0, 0.05), (-0.1, 0.08, 0.0)):
    lam, mode = _t_call(1, _t_s, _t_u, _t_v)
    assert _t_rel(lam, 1.0 - _t_u - _t_v) < 1e-8, (_t_s, _t_u, _t_v, lam)
    _t_x = np.array([_t_v, -_t_u]) / max(_t_u, _t_v)
    _t_check_mode(mode, _t_x if _t_v > 0 else -_t_x, _t_absorbing_mask(1, _t_u, _t_v), 1e-8, (1, _t_s, _t_u, _t_v))

# --- test case 1 ---
# neutral drift: the eigenvalues are (1 - u - v)**k N! / (N**k (N - k)!), so the smallest is
# (1 - u - v)**N N! / N**N, about 1e-173 at N = 400 with rare or no mutations and 1.7e-211 with u = v = 0.1;
# the mode against extended precision at moderate N
import numpy as np
import mpmath as _t_mp
import math as _t_math

def _t_P(N, s, u, v):
    # call inside _t_mp.workdps(...)
    s, u, v = _t_mp.mpf(s), _t_mp.mpf(u), _t_mp.mpf(v)
    P = _t_mp.matrix(N + 1, N + 1)
    for i in range(N + 1):
        p = _t_mp.mpf(i) / N
        ps = (1 + s) * p / (1 + s * p)
        pm = (1 - u) * ps + v * (1 - ps)
        for j in range(N + 1):
            P[i, j] = _t_mp.binomial(N, j) * pm ** j * (1 - pm) ** (N - j)
    return P

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_call(N, s, u, v):
    out = fastest_mode(N, s, u, v)
    lam, mode = out
    assert isinstance(lam, float), lam
    mode = np.asarray(mode)
    assert mode.shape == (N + 1,) and np.issubdtype(mode.dtype, np.floating), (mode.shape, mode.dtype)
    return lam, mode

def _t_absorbing_mask(N, u, v):
    # mask of the absorbing states (i = 0 when v = 0, i = N when u = 0), where the exact mode is zero
    zero = np.zeros(N + 1, dtype=bool)
    zero[0] = not v > 0
    zero[N] = zero[N] or not u > 0
    return zero

def _t_check_mode(got, target, zero, tol, label):
    # entries whose exact value is zero (absorbing states) within 1e-250 of zero; every other entry within
    # tol times its magnitude (signs and the normalisation as stated)
    got = np.asarray(got, dtype=float)
    target = np.asarray(target, dtype=float)
    zero = np.asarray(zero, dtype=bool)
    assert np.all(np.abs(got[zero]) <= 1e-250), (label, got[zero])
    bad = ~zero & (np.abs(got - target) > tol * np.abs(target))
    assert not bad.any(), (label, np.nonzero(bad)[0][:5], got[bad][:5], target[bad][:5])

def _t_mp_mode(N, s, u, v, dps):
    # smallest eigenvalue from mpmath eig; eigenvector by shifted inverse iteration with plain LU,
    # started with zeros on the absorbing states; scaled to max |entry| = 1, first nonzero entry positive
    with _t_mp.workdps(dps):
        P = _t_P(N, s, u, v)
        E = _t_mp.eig(P, left=False, right=False)
        lam = min(E, key=lambda z: abs(z)).real
        n = N + 1
        A = P - lam * (1 + _t_mp.mpf(10) ** (-dps // 3)) * _t_mp.eye(n)
        x = _t_mp.matrix([0 if P[i, i] == 1 else 1 for i in range(n)])
        for _ in range(8):
            x = _t_mp.lu_solve(A, x)
            x = x / max(abs(t) for t in x)
        first = [t for t in x if t != 0][0]
        x = x / _t_mp.sign(first)
        return float(lam), [float(t) for t in x]

for N, u, v in ((2, 0.1, 0.1), (50, 1e-3, 0.02), (400, 1e-12, 1e-12), (400, 0.1, 0.1), (333, 1e-7, 0.05),
                (400, 0.0, 0.0), (250, 0.0, 0.03)):
    exact = _t_math.exp(N * _t_math.log1p(-(u + v)) + _t_math.lgamma(N + 1) - N * _t_math.log(N))
    got, mode = _t_call(N, 0.0, u, v)
    assert _t_rel(got, exact) < 1e-8, (N, u, v, got, exact)
    # the mode vanishes on the absorbing states (exact value 0, required within 1e-250)
    zero = _t_absorbing_mask(N, u, v)
    assert np.all(np.abs(mode[zero]) <= 1e-250), (N, u, v, mode[zero])
for N, u, v in ((10, 0.0, 0.0), (17, 0.02, 1e-9)):
    got, mode = _t_call(N, 0.0, u, v)
    lam_t, mode_t = _t_mp_mode(N, 0.0, u, v, 100)
    _t_check_mode(mode, mode_t, _t_absorbing_mask(N, u, v), 1e-8, (N, 0.0, u, v))

# --- test case 2 ---
# selection, with and without absorbing states, against the matrix in extended precision
import numpy as np
import mpmath as _t_mp

def _t_P(N, s, u, v):
    # call inside _t_mp.workdps(...)
    s, u, v = _t_mp.mpf(s), _t_mp.mpf(u), _t_mp.mpf(v)
    P = _t_mp.matrix(N + 1, N + 1)
    for i in range(N + 1):
        p = _t_mp.mpf(i) / N
        ps = (1 + s) * p / (1 + s * p)
        pm = (1 - u) * ps + v * (1 - ps)
        for j in range(N + 1):
            P[i, j] = _t_mp.binomial(N, j) * pm ** j * (1 - pm) ** (N - j)
    return P

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_call(N, s, u, v):
    out = fastest_mode(N, s, u, v)
    lam, mode = out
    assert isinstance(lam, float), lam
    mode = np.asarray(mode)
    assert mode.shape == (N + 1,) and np.issubdtype(mode.dtype, np.floating), (mode.shape, mode.dtype)
    return lam, mode

def _t_absorbing_mask(N, u, v):
    # mask of the absorbing states (i = 0 when v = 0, i = N when u = 0), where the exact mode is zero
    zero = np.zeros(N + 1, dtype=bool)
    zero[0] = not v > 0
    zero[N] = zero[N] or not u > 0
    return zero

def _t_check_mode(got, target, zero, tol, label):
    # entries whose exact value is zero (absorbing states) within 1e-250 of zero; every other entry within
    # tol times its magnitude (signs and the normalisation as stated)
    got = np.asarray(got, dtype=float)
    target = np.asarray(target, dtype=float)
    zero = np.asarray(zero, dtype=bool)
    assert np.all(np.abs(got[zero]) <= 1e-250), (label, got[zero])
    bad = ~zero & (np.abs(got - target) > tol * np.abs(target))
    assert not bad.any(), (label, np.nonzero(bad)[0][:5], got[bad][:5], target[bad][:5])

def _t_mp_mode(N, s, u, v, dps):
    # smallest eigenvalue from mpmath eig; eigenvector by shifted inverse iteration with plain LU,
    # started with zeros on the absorbing states; scaled to max |entry| = 1, first nonzero entry positive
    with _t_mp.workdps(dps):
        P = _t_P(N, s, u, v)
        E = _t_mp.eig(P, left=False, right=False)
        lam = min(E, key=lambda z: abs(z)).real
        n = N + 1
        A = P - lam * (1 + _t_mp.mpf(10) ** (-dps // 3)) * _t_mp.eye(n)
        x = _t_mp.matrix([0 if P[i, i] == 1 else 1 for i in range(n)])
        for _ in range(8):
            x = _t_mp.lu_solve(A, x)
            x = x / max(abs(t) for t in x)
        first = [t for t in x if t != 0][0]
        x = x / _t_mp.sign(first)
        return float(lam), [float(t) for t in x]

for N, s, u, v, dps in ((12, 0.5, 0.01, 0.02, 80), (20, -0.5, 1e-12, 1e-12, 120), (16, 0.3, 0.0, 0.05, 100),
                        (15, -0.2, 0.05, 0.0, 100), (14, 0.4, 0.0, 0.0, 100), (2, -0.3, 0.0, 0.0, 40),
                        (25, 0.45, 1e-12, 0.0, 120), (30, -0.5, 0.1, 0.1, 150)):
    got, mode = _t_call(N, s, u, v)
    lam_t, mode_t = _t_mp_mode(N, s, u, v, dps)
    assert _t_rel(got, lam_t) < 1e-8, (N, s, u, v, got, lam_t)
    _t_check_mode(mode, mode_t, _t_absorbing_mask(N, u, v), 1e-8, (N, s, u, v))

# --- test case 3 ---
# N = 400 with strong selection: smallest eigenvalues between 1e-197 and 1e-174 and modes
# whose entries span up to 176 decades, against inverse iteration with plain LU in mpmath at 400 digits
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_call(N, s, u, v):
    out = fastest_mode(N, s, u, v)
    lam, mode = out
    assert isinstance(lam, float), lam
    mode = np.asarray(mode)
    assert mode.shape == (N + 1,) and np.issubdtype(mode.dtype, np.floating), (mode.shape, mode.dtype)
    return lam, mode

def _t_absorbing_mask(N, u, v):
    # mask of the absorbing states (i = 0 when v = 0, i = N when u = 0), where the exact mode is zero
    zero = np.zeros(N + 1, dtype=bool)
    zero[0] = not v > 0
    zero[N] = zero[N] or not u > 0
    return zero

def _t_check_mode(got, target, zero, tol, label):
    # entries whose exact value is zero (absorbing states) within 1e-250 of zero; every other entry within
    # tol times its magnitude (signs and the normalisation as stated)
    got = np.asarray(got, dtype=float)
    target = np.asarray(target, dtype=float)
    zero = np.asarray(zero, dtype=bool)
    assert np.all(np.abs(got[zero]) <= 1e-250), (label, got[zero])
    bad = ~zero & (np.abs(got - target) > tol * np.abs(target))
    assert not bad.any(), (label, np.nonzero(bad)[0][:5], got[bad][:5], target[bad][:5])

# (run once; the stored entries, as "index:value", are i = 0, 8, 16, ..., 400 and the entries next to an
# absorbing state)
_t_big = (
    ((400, 0.5, 1e-12, 1e-12), "2.76656467220724976158637404511e-176",
     "0:5.29897091091137642e-36 8:4.33758730791651745e-22 16:2.13048054873020246e-18 "
     "24:2.41565728306455263e-15 32:9.87682824415069916e-13 40:1.78892126390238676e-10 "
     "48:1.62108667104663668e-08 56:7.97180083534954595e-07 64:2.25525534356617128e-05 "
     "72:3.83567637811313227e-04 80:4.05893361083170708e-03 88:2.74687387488468174e-02 "
     "96:1.21575803416687778e-01 104:3.58486415371397527e-01 112:7.15190781654126306e-01 "
     "120:9.77981468455975222e-01 128:9.26731092507446230e-01 136:6.14196302257519755e-01 "
     "144:2.86930096827253178e-01 152:9.51037715020496000e-02 160:2.24867067205769718e-02 "
     "168:3.80960946567146823e-03 176:4.64072808477666810e-04 184:4.07568877652243847e-05 "
     "192:2.58546162072419387e-06 200:1.18598614901831550e-07 208:3.93536971536141124e-09 "
     "216:9.44270236304247673e-11 224:1.63655289943137269e-12 232:2.04491000780318063e-14 "
     "240:1.83726147072553064e-16 248:1.18275862566624845e-18 256:5.43162771540466328e-21 "
     "264:1.76977940055853138e-23 272:4.06468167674387600e-26 280:6.52931119401689181e-29 "
     "288:7.26818339921101580e-32 296:5.54558516246292421e-35 304:2.86283403214558282e-38 "
     "312:9.84613669978392733e-42 320:2.21473152044684249e-45 328:3.18593603439243733e-49 "
     "336:2.85154867712336991e-53 344:1.53438925565181993e-57 352:4.74988925117847778e-62 "
     "360:7.97940942809975149e-67 368:6.70661661907987061e-72 376:2.49707749161373334e-77 "
     "384:3.35265937256259010e-83 392:1.03915441611047566e-89 400:4.34843891804748381e-106 "),
    ((400, -0.5, 0.1, 1e-12), "5.87144975069309365989907283059e-197",
     "0:3.26568009878679104e-176 8:1.32279824451308375e-158 16:1.12024189656626594e-150 "
     "24:2.24893770961475451e-143 32:1.66655780762679605e-136 40:5.59215893043726367e-130 "
     "48:9.58950599087067887e-124 56:9.11239386832256580e-118 64:5.08654367255551808e-112 "
     "72:1.74319420890229473e-106 80:3.79700417503545676e-101 88:5.40525260294156801e-96 "
     "96:5.14535064747149189e-91 104:3.33837280049530683e-86 112:1.50031876483872166e-81 "
     "120:4.73519823418563234e-77 128:1.06202616378557640e-72 136:1.71007150522182096e-68 "
     "144:1.99447389068495361e-64 152:1.69797046818408284e-60 160:1.06228262763444264e-56 "
     "168:4.91248258307008485e-53 176:1.68780456167100657e-49 184:4.32727797042919679e-46 "
     "192:8.31039337128699265e-43 200:1.19933470712722348e-39 208:1.30419924759201958e-36 "
     "216:1.07101532296562963e-33 224:6.65369382177067858e-31 232:3.13131381043518223e-28 "
     "240:1.11735315847309602e-25 248:3.02472666236391008e-23 256:6.21266533352690060e-21 "
     "264:9.67978314883057762e-19 272:1.14336634092601542e-16 280:1.02284524484939670e-14 "
     "288:6.92064717812754969e-13 296:3.53537314092036136e-11 304:1.36063932600715274e-09 "
     "312:3.93513840202956335e-08 320:8.52692164879915675e-07 328:1.37960294180191899e-05 "
     "336:1.66019554876304406e-04 344:1.47948616531236527e-03 352:9.71608111689009797e-03 "
     "360:4.67680717736453991e-02 368:1.64018037508649572e-01 376:4.16359715422115695e-01 "
     "384:7.59570847181760822e-01 392:9.88117667702938851e-01 400:9.08947763715388435e-01 "),
    ((400, 0.3, 0.0, 0.0), "3.12147827485774064785295459392e-174",
     "0:0 1:5.60473308396838027e-35 8:-4.99702223225142394e-30 16:-7.71135673178739542e-26 "
     "24:-2.74713751610333131e-22 32:-3.52903522893220707e-19 40:-2.00828167954900054e-16 "
     "48:-5.71789285968926953e-14 56:-8.83454230513634262e-12 64:-7.85276538104641352e-10 "
     "72:-4.19632956854980018e-08 80:-1.39521694130832840e-06 88:-2.96668947040761601e-05 "
     "96:-4.12558454435142012e-04 104:-3.82223496227175434e-03 112:-2.39593064694339740e-02 "
     "120:-1.02941831157481192e-01 128:-3.06496231042712397e-01 136:-6.38248615889166038e-01 "
     "144:-9.36852296743106328e-01 152:-9.75676985516628736e-01 160:-7.24850321618247917e-01 "
     "168:-3.85849411751986671e-01 176:-1.47685900830220790e-01 184:-4.07539726457843213e-02 "
     "192:-8.12313014420214079e-03 200:-1.17079680742269185e-03 208:-1.22068644309756546e-04 "
     "216:-9.20304362288906852e-06 224:-5.01166010205212908e-07 232:-1.96762251680744176e-08 "
     "240:-5.55461645756279188e-10 248:-1.12355446281885477e-11 256:-1.62122070603035203e-13 "
     "264:-1.65975715944022065e-15 272:-1.19774465919093077e-17 280:-6.04527805921054356e-20 "
     "288:-2.11438873173891147e-22 296:-5.06891653571077327e-25 304:-8.22188556822241296e-28 "
     "312:-8.88479170968447150e-31 320:-6.27924172797661180e-34 328:-2.83809187033467623e-37 "
     "336:-7.98126990587375612e-41 344:-1.34935764497043365e-44 352:-1.31242294459925739e-48 "
     "360:-6.92721005467398303e-53 368:-1.82930792040011924e-57 376:-2.13997892026622038e-62 "
     "384:-9.02733160905931510e-68 392:-8.79104967239994950e-74 399:2.50422884786416618e-80 "
     "400:0 "),
    ((400, -0.4, 0.0, 0.05), "5.01106891557718685161053070304e-190",
     "0:1.21732693567125595e-66 8:6.66924987892986146e-63 16:2.82712252596809334e-59 "
     "24:9.28889769283126431e-56 32:2.36947502410552312e-52 40:4.69979619343979022e-49 "
     "48:7.25875894147427505e-46 56:8.74099422884070966e-43 64:8.21616659278517893e-40 "
     "72:6.03412591549339974e-37 80:3.46538324513719836e-34 88:1.55725566132560852e-31 "
     "96:5.47825338460275506e-29 104:1.50910167058278927e-26 112:3.25555859069225651e-24 "
     "120:5.49936143843160267e-22 128:7.27171941529610653e-20 136:7.52253076153228501e-18 "
     "144:6.08356459781813139e-16 152:3.84221353486383228e-14 160:1.89272393437253622e-12 "
     "168:7.26128831729040371e-11 176:2.16559296457704936e-09 184:5.01028494915900601e-08 "
     "192:8.97048885159823187e-07 200:1.23945922237619232e-05 208:1.31747006198670630e-04 "
     "216:1.07347451562091862e-03 224:6.67785963150551928e-03 232:3.15725738819124316e-02 "
     "240:1.12874889931066827e-01 248:3.03396851866568285e-01 256:6.09186298385181901e-01 "
     "264:9.07100944331568848e-01 272:9.93471571201517190e-01 280:7.92858126535629482e-01 "
     "288:4.56197628737667005e-01 296:1.86950340664944092e-01 304:5.38010638545528136e-02 "
     "312:1.06957247269624956e-02 320:1.44071682628518251e-03 328:1.28488542411681495e-04 "
     "336:7.37730195263849255e-06 344:2.63393509953143943e-07 352:5.59514694649852743e-09 "
     "360:6.67166869641460086e-11 368:4.11940833898598429e-13 376:1.16763341760336541e-15 "
     "384:1.23980330694987258e-18 392:3.17369923527369027e-22 399:-1.24183356071163491e-26 400:0 "),
)
for (N, s, u, v), lam_t, mode_t in _t_big:
    got, mode = _t_call(N, s, u, v)
    assert _t_rel(got, float(lam_t)) < 1e-8, (N, s, u, v, got, lam_t)
    idx = np.array([int(t.split(":")[0]) for t in mode_t.split()])
    val = np.array([float(t.split(":")[1]) for t in mode_t.split()])
    _t_check_mode(mode[idx], val, _t_absorbing_mask(N, u, v)[idx], 1e-8, (N, s, u, v))

# --- test case 4 ---
# relabeling the alleles (s' = -s / (1 + s), u and v exchanged) reverses the states: same
# eigenvalue, mode reversed (up to the sign fixed by the normalisation), absorbing state moved to the other end
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_call(N, s, u, v):
    out = fastest_mode(N, s, u, v)
    lam, mode = out
    assert isinstance(lam, float), lam
    mode = np.asarray(mode)
    assert mode.shape == (N + 1,) and np.issubdtype(mode.dtype, np.floating), (mode.shape, mode.dtype)
    return lam, mode

def _t_absorbing_mask(N, u, v):
    # mask of the absorbing states (i = 0 when v = 0, i = N when u = 0), where the exact mode is zero
    zero = np.zeros(N + 1, dtype=bool)
    zero[0] = not v > 0
    zero[N] = zero[N] or not u > 0
    return zero

def _t_check_mode(got, target, zero, tol, label):
    # entries whose exact value is zero (absorbing states) within 1e-250 of zero; every other entry within
    # tol times its magnitude (signs and the normalisation as stated)
    got = np.asarray(got, dtype=float)
    target = np.asarray(target, dtype=float)
    zero = np.asarray(zero, dtype=bool)
    assert np.all(np.abs(got[zero]) <= 1e-250), (label, got[zero])
    bad = ~zero & (np.abs(got - target) > tol * np.abs(target))
    assert not bad.any(), (label, np.nonzero(bad)[0][:5], got[bad][:5], target[bad][:5])

for (N, s, u, v) in ((400, 0.3, 0.0, 0.05), (400, 0.3, 1e-6, 0.05), (77, -0.3, 0.0, 0.0)):
    la, ma = _t_call(N, s, u, v)
    lb, mb = _t_call(N, -s / (1.0 + s), v, u)
    assert _t_rel(la, lb) < 2.1e-8, (N, s, u, v, la, lb)   # two computed outputs, each within 1e-8: 2.1e-8
    zero = _t_absorbing_mask(N, u, v)
    assert np.all(np.abs(mb[::-1][zero]) <= 1e-250), (N, s, u, v)
    # sign fixed by the first entry that is not on an absorbing state
    rb = mb[::-1] * np.sign(mb[::-1][np.nonzero(~zero)[0][0]])
    _t_check_mode(ma, rb, zero, 2.1e-8, ("relabel", N, s, u, v))

# --- test case 5 ---
# N not an integer in [1, 400], s, u, v not finite or out of range (u, v >= 0), or N = 1
# without mutation
import numpy as np
for _t_bad in ((0, 0.1, 0.01, 0.01), (401, 0.1, 0.01, 0.01), (5.0, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01),
               (10, 0.6, 0.01, 0.01), (10, -0.6, 0.01, 0.01), (10, 0.1, -1e-12, 0.01), (10, 0.1, 0.01, -1e-12),
               (10, 0.1, 0.2, 0.01), (10, 0.1, 0.01, 0.11), (10, float("nan"), 0.01, 0.01),
               (10, 0.1, float("inf"), 0.01), (1, 0.2, 0.0, 0.0)):
    try:
        fastest_mode(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("fastest_mode%r must raise ValueError" % (_t_bad,))
