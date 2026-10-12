# --- test case 0 ---
# Independent targets: the exact neutral spectrum, the Wright-Fisher matrix built in extended precision
# (mpmath binomials) with its eigenvalues from mpmath eig and the eigenvector by inverse iteration with a
# plain LU, and for N = 400 inverse iteration with a plain LU in mpmath at 400 digits (run once); no use
# of the Vandermonde structure.
# one individual: the 2 x 2 matrix has the eigenvalues 1 and 1 - u - v with the right
# eigenvector (v, -u) (sign fixed by the normalisation); with u = 0 or v = 0 one state is absorbing and the
# mode vanishes there; with small u + v the two eigenvalues are close to each other (ratio 1 - u - v)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_call(N, s, u, v):
        # the return type: (Python float, numpy float array of shape (N + 1,))
        lam, mode = fastest_mode(N, s, u, v)
        mode = np.asarray(mode)
        assert isinstance(lam, float) and mode.shape == (N + 1,) and np.issubdtype(mode.dtype, np.floating), (lam, mode.shape, mode.dtype)
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

    for _t_s, _t_u, _t_v in ((0.3, 0.05, 0.02), (-0.5, 1e-12, 0.1), (0.2, 0.0, 0.05), (-0.1, 0.08, 0.0),
                             (0.2, 1e-6, 1e-12), (-0.4, 2e-12, 5e-12)):
        lam, mode = _t_call(1, _t_s, _t_u, _t_v)
        assert _t_rel(lam, 1.0 - _t_u - _t_v) < 1e-8, (_t_s, _t_u, _t_v, lam)
        _t_x = np.array([_t_v, -_t_u]) / max(_t_u, _t_v)
        _t_check_mode(mode, _t_x if _t_v > 0 else -_t_x, _t_absorbing_mask(1, _t_u, _t_v), 1e-8, (1, _t_s, _t_u, _t_v))


_crown_case()

# --- test case 1 ---
# neutral drift: the eigenvalues are (1 - u - v)**k N! / (N**k (N - k)!), so the smallest is
# (1 - u - v)**N N! / N**N, about 9.6e-173 at N = 400 with rare or no mutations and 1.7e-211 with u = v = 0.1;
# the mode against extended precision at moderate N
import numpy as np


def _crown_case():
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
        # the return type: (Python float, numpy float array of shape (N + 1,))
        lam, mode = fastest_mode(N, s, u, v)
        mode = np.asarray(mode)
        assert isinstance(lam, float) and mode.shape == (N + 1,) and np.issubdtype(mode.dtype, np.floating), (lam, mode.shape, mode.dtype)
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


_crown_case()

# --- test case 2 ---
# selection with mutation in both directions or one way (one absorbing state), against the matrix in extended
# precision (both end states absorbing: case 6)
import numpy as np


def _crown_case():
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
        # values only (the return type, shape and dtype are checked in cases 0 and 1)
        lam, mode = fastest_mode(N, s, u, v)
        return lam, np.asarray(mode, dtype=float)

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
                            (15, -0.2, 0.05, 0.0, 100)):
        got, mode = _t_call(N, s, u, v)
        lam_t, mode_t = _t_mp_mode(N, s, u, v, dps)
        assert _t_rel(got, lam_t) < 1e-8, (N, s, u, v, got, lam_t)
        _t_check_mode(mode, mode_t, _t_absorbing_mask(N, u, v), 1e-8, (N, s, u, v))


_crown_case()

# --- test case 3 ---
# N = 400 with strong selection: smallest eigenvalues between 1e-197 and 1e-174 and modes
# whose entries span up to 175.5 decades, against inverse iteration with plain LU in mpmath at 400 digits
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_call(N, s, u, v):
        # values only (the return type, shape and dtype are checked in cases 0 and 1)
        lam, mode = fastest_mode(N, s, u, v)
        return lam, np.asarray(mode, dtype=float)

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

    # (run once; the stored entries, as "index:value", are i = 0, 5, 10, ..., 400, odd and even indices so that
    # the alternating signs are checked, and the entries next to an absorbing state)
    _t_big = (
        ((400, 0.5, 1e-12, 1e-12), "2.76656467220724976158637404511e-176",
         "0:5.29897091091137642e-36 5:-9.59689361060589911e-24 10:4.36838980179877080e-21 15:-8.09314434431204430e-19 "
         "20:8.30275776888718094e-17 25:-5.38686218722446837e-15 30:2.38510800071750666e-13 "
         "35:-7.57600682694302700e-12 40:1.78892126390238676e-10 45:-3.22541171081863739e-09 "
         "50:4.53427378004368476e-08 55:-5.05438074918052431e-07 60:4.52979021225340588e-06 "
         "65:-3.30196060747658783e-05 70:1.97708048478929555e-04 75:-9.80638057297775721e-04 "
         "80:4.05893361083170708e-03 85:-1.41098102862622019e-02 90:4.14276695968204753e-02 "
         "95:-1.03250654175705725e-01 100:2.19412267941842182e-01 105:-3.99135672520692319e-01 "
         "110:6.23759587352386347e-01 115:-8.40113981948077959e-01 120:9.77981468455975222e-01 "
         "125:-9.86544489764599852e-01 130:8.64383230255562740e-01 135:-6.59186125474809592e-01 "
         "140:4.38367031120439399e-01 145:-2.54640302845337119e-01 150:1.29398440163036255e-01 "
         "155:-5.76002388366859408e-02 160:2.24867067205769718e-02 165:-7.70695982419234045e-03 "
         "170:2.32105347942042611e-03 175:-6.14700920876134387e-04 180:1.43250522411509653e-04 "
         "185:-2.93904513537787825e-05 190:5.31088251475465822e-06 195:-8.45473647250195011e-07 "
         "200:1.18598614901831550e-07 205:-1.46598559364278040e-08 210:1.59671000249367964e-09 "
         "215:-1.53212968505806147e-10 220:1.29483877776377839e-11 225:-9.63416198511044177e-13 "
         "230:6.30763137520808988e-14 235:-3.63158238074810293e-15 240:1.83726147072553064e-16 "
         "245:-8.16023087963621572e-18 250:3.17863806558573293e-19 255:-1.08461089251084280e-20 "
         "260:3.23757966032021783e-22 265:-8.44165200268363591e-24 270:1.91939183436701598e-25 "
         "275:-3.79850834570068449e-27 280:6.52931119401689181e-29 285:-9.72560380796462090e-31 "
         "290:1.25209864893845596e-32 295:-1.38926849130132966e-34 300:1.32425704532663605e-36 "
         "305:-1.08056623101160173e-38 310:7.51792097817593937e-41 315:-4.43995538501803327e-43 "
         "320:2.21473152044684249e-45 325:-9.27835822998920888e-48 330:3.24370655385283207e-50 "
         "335:-9.39389039165599646e-53 340:2.23465080129300778e-55 345:-4.32373537360558355e-58 "
         "350:6.72604962058570041e-61 355:-8.29660231909211159e-64 360:7.97940942809975149e-67 "
         "365:-5.85984220558585969e-70 370:3.19905203514188649e-73 375:-1.25290599571249963e-76 "
         "380:3.34869871594488924e-80 385:-5.66035756901658455e-84 390:5.29813166933009140e-88 "
         "395:-2.01840811874903647e-92 400:4.34843891804748381e-106 "),
        ((400, -0.5, 0.1, 1e-12), "5.87144975069309365989907283059e-197",
         "0:3.26568009878679104e-176 5:-7.60755769067002465e-162 10:1.52230302277826908e-156 "
         "15:-1.25517968561761321e-151 20:5.79138094419599871e-147 25:-1.70617131186065431e-142 "
         "30:3.46126941575134981e-138 35:-5.08122102995151446e-134 40:5.59215893043726367e-130 "
         "45:-4.73838968704526677e-126 50:3.15623953249581532e-122 55:-1.68069137421181919e-118 "
         "60:7.25428038412846559e-115 65:-2.56763034342870783e-111 70:7.52657967708300752e-108 "
         "75:-1.84286032286527726e-104 80:3.79700417503545676e-101 85:-6.62636960700692768e-98 "
         "90:9.85149019399587053e-95 95:-1.25415426395156804e-91 100:1.37347262150401927e-88 "
         "105:-1.29928792193588625e-85 110:1.06568603970458816e-82 115:-7.60434042368071800e-80 "
         "120:4.73519823418563234e-77 125:-2.58033652174857678e-74 130:1.23363730849149246e-71 "
         "135:-5.18666366271354039e-69 140:1.92180376609661754e-66 145:-6.28787256031760086e-64 "
         "150:1.81993099553143860e-61 155:-4.66746180665053054e-59 160:1.06228262763444264e-56 "
         "165:-2.14850734499810664e-54 170:3.86655580922677657e-52 175:-6.19881050727494558e-50 "
         "180:8.86239041382816581e-48 185:-1.13102503214302954e-45 190:1.28959621878085734e-43 "
         "195:-1.31474030896999747e-41 200:1.19933470712722348e-39 205:-9.79560328634901388e-38 "
         "210:7.16731722881161698e-36 215:-4.70035228401405031e-34 220:2.76399175473119327e-32 "
         "225:-1.45791205222589271e-30 230:6.89989011100798501e-29 235:-2.93069302750795487e-27 "
         "240:1.11735315847309602e-25 245:-3.82429452165795914e-24 250:1.17510370482699571e-22 "
         "255:-3.24162727876788872e-21 260:8.02758607732921324e-20 265:-1.78439064627919904e-18 "
         "270:3.55959951844940939e-17 275:-6.37113447655051482e-16 280:1.02284524484939670e-14 "
         "285:-1.47240875544675878e-13 290:1.89974066154903363e-12 295:-2.19585619614598018e-11 "
         "300:2.27262029954351050e-10 305:-2.10477702912344006e-09 310:1.74323855887672916e-08 "
         "315:-1.29022710075963566e-07 320:8.52692164879915675e-07 325:-5.02763340496419919e-06 "
         "330:2.64227491475286036e-05 335:-1.23652185820767927e-04 340:5.14717363208795978e-04 "
         "345:-1.90361605073031200e-03 350:6.24739704980260043e-03 355:-1.81701315285349510e-02 "
         "360:4.67680717736453991e-02 365:-1.06372324957580380e-01 370:2.13457847184051558e-01 "
         "375:-3.77291124832593749e-01 380:5.86351708917545955e-01 385:-7.99745855595748401e-01 "
         "390:9.55458273825463889e-01 395:-9.97814859811412824e-01 400:9.08947763715388435e-01 "),
        ((400, 0.3, 0.0, 0.0), "3.12147827485774064785295459392e-174",
         "0:0 1:5.60473308396838027e-35 5:7.19694068580431547e-32 10:-6.70010804435590828e-29 "
         "15:2.53875995353439672e-26 20:-5.32686043722757245e-24 25:7.06854518552023627e-22 "
         "30:-6.40100323578193714e-20 35:4.15840482936813360e-18 40:-2.00828167954900054e-16 "
         "45:7.40570014671294400e-15 50:-2.12929993360826722e-13 55:4.85451402676306962e-12 "
         "60:-8.89826153811514911e-11 65:1.32662690814845079e-09 70:-1.62461950642698669e-08 "
         "75:1.64811312685554903e-07 80:-1.39521694130832840e-06 85:9.91980281228670532e-06 "
         "90:-5.95696417943938190e-05 95:3.03655015218245777e-04 100:-1.31978214594445230e-03 "
         "105:4.91039136961978796e-03 110:-1.56952245301092454e-02 115:4.32358249418761179e-02 "
         "120:-1.02941831157481192e-01 125:2.12390036310205782e-01 130:-3.80610226531538309e-01 "
         "135:5.93662029433927207e-01 140:-8.07470398363321240e-01 145:9.59342699379200581e-01 "
         "150:-9.97089062396307901e-01 155:9.07795070385845282e-01 160:-7.24850321618247917e-01 "
         "165:5.08118270453311260e-01 170:-3.12987044508261425e-01 175:1.69537122158464781e-01 "
         "180:-8.08085076046445167e-02 185:3.39099557125280768e-02 190:-1.25327936553082508e-02 "
         "195:4.08076511067192484e-03 200:-1.17079680742269185e-03 205:2.96000220215257364e-04 "
         "210:-6.59399767828223576e-05 215:1.29413219362305988e-05 220:-2.23696455948121954e-06 "
         "225:3.40421930622782391e-07 230:-4.55858360125496362e-08 235:5.36808484484354407e-09 "
         "240:-5.55461645756279188e-10 245:5.04597329272694536e-11 250:-4.02015405515897877e-12 "
         "255:2.80565576899857773e-13 260:-1.71292899003606738e-14 265:9.13491375735097400e-16 "
         "270:-4.24813560519654535e-17 275:1.71951165275441281e-18 280:-6.04527805921054356e-20 "
         "285:1.84171280154836333e-21 290:-4.84953195160358411e-23 295:1.10053261344086424e-24 "
         "300:-2.14557632410803941e-26 305:3.58077877081970218e-28 310:-5.09539367204269299e-30 "
         "315:6.15477151482440231e-32 320:-6.27924172797661180e-34 325:5.38033694357432197e-36 "
         "330:-3.84708292276323581e-38 335:2.27869684837083721e-40 340:-1.10866725881739779e-42 "
         "345:4.38733265683049602e-45 350:-1.39588885208213013e-47 355:3.52159787527109473e-50 "
         "360:-6.92721005467398303e-53 365:1.04044958526125443e-55 370:-1.16172496981257537e-58 "
         "375:9.30565149416980629e-62 380:-5.08686261579406820e-65 385:1.75858259319418740e-68 "
         "390:-3.36655650530458749e-72 395:2.62310452197711267e-76 399:2.50422884786416618e-80 400:0 "),
        ((400, -0.4, 0.0, 0.05), "5.01106891557718685161053070304e-190",
         "0:1.21732693567125595e-66 5:-2.72378142822194759e-64 10:5.51198264997190739e-62 15:-1.00925842607373908e-59 "
         "20:1.67278316702329891e-57 25:-2.51071740839606303e-55 30:3.41388358611720861e-53 "
         "35:-4.20684838430046859e-51 40:4.69979619343979022e-49 45:-4.76171586221072695e-47 "
         "50:4.37672734560437800e-45 55:-3.65062200389113810e-43 60:2.76398370047354752e-41 "
         "65:-1.90004936482887965e-39 70:1.18618920509349804e-37 75:-6.72652232413835758e-36 "
         "80:3.46538324513719836e-34 85:-1.62218695852405262e-32 90:6.90068891735989437e-31 "
         "95:-2.66788219496619330e-29 100:9.37457844084350278e-28 105:-2.99407865455729274e-26 "
         "110:8.69167342626227546e-25 115:-2.29330239470302971e-23 120:5.49936143843160267e-22 "
         "125:-1.19844321036322373e-20 130:2.37314210365802587e-19 135:-4.26936376711901211e-18 "
         "140:6.97673992742049594e-17 145:-1.03536392786823377e-15 150:1.39498537676479207e-14 "
         "155:-1.70588882308373902e-13 160:1.89272393437253622e-12 165:-1.90463792545519655e-11 "
         "170:1.73756352495593648e-10 175:-1.43637284981597067e-09 180:1.07539203352795270e-08 "
         "185:-7.28774757502437900e-08 190:4.46764120832564435e-07 195:-2.47588236359421220e-06 "
         "200:1.23945922237619232e-05 205:-5.60070730467098155e-05 210:2.28240329541331042e-04 "
         "215:-8.38073632994756706e-04 220:2.77001890144512380e-03 225:-8.23249625499761391e-03 "
         "230:2.19750972824120892e-02 235:-5.26192761313993979e-02 240:1.12874889931066827e-01 "
         "245:-2.16604693287347855e-01 250:3.71268820710879854e-01 255:-5.67465565239971492e-01 "
         "260:7.72049077779001003e-01 265:-9.33183202193981542e-01 270:1.00000000000000000e+00 "
         "275:-9.47902982677332417e-01 280:7.92858126535629482e-01 285:-5.83632533480259008e-01 "
         "290:3.76996672689609902e-01 295:-2.13017442585045486e-01 300:1.04922066658828278e-01 "
         "305:-4.48784166791062616e-02 310:1.65997271109407270e-02 315:-5.28485301837888950e-03 "
         "320:1.44071682628518251e-03 325:-3.34358239376563154e-04 330:6.56272968859817821e-05 "
         "335:-1.08133268975803344e-05 340:1.48294855732375366e-06 345:-1.67605058664313140e-07 "
         "350:1.54313122206970065e-08 355:-1.14149832051248285e-09 360:6.67166869641460086e-11 "
         "365:-3.01762785694209987e-12 370:1.02862621834102108e-13 375:-2.55108235958350195e-15 "
         "380:4.38154093592301930e-17 385:-4.83417581210534173e-19 390:3.00462937615758515e-21 "
         "395:-7.75580314503175010e-24 399:-1.24183356071163491e-26 400:0 "),
    )
    for (N, s, u, v), lam_t, mode_t in _t_big:
        got, mode = _t_call(N, s, u, v)
        assert _t_rel(got, float(lam_t)) < 1e-8, (N, s, u, v, got, lam_t)
        idx = np.array([int(t.split(":")[0]) for t in mode_t.split()])
        val = np.array([float(t.split(":")[1]) for t in mode_t.split()])
        _t_check_mode(mode[idx], val, _t_absorbing_mask(N, u, v)[idx], 1e-8, (N, s, u, v))


_crown_case()

# --- test case 4 ---
# relabeling the alleles (s' = -s / (1 + s), u and v exchanged) reverses the states: same
# eigenvalue, mode reversed (up to the sign fixed by the normalisation), absorbing state moved to the other end
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_call(N, s, u, v):
        # values only (the return type, shape and dtype are checked in cases 0 and 1)
        lam, mode = fastest_mode(N, s, u, v)
        return lam, np.asarray(mode, dtype=float)

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


_crown_case()

# --- test case 5 ---
# N not an integer in [1, 400], s, u, v out of range (u and v each 0 or in [1e-12, 0.1], so a positive rate
# below 1e-12 is out of range, however small), or N = 1 without mutation (P is the identity)
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((0, 0.1, 0.01, 0.01), (401, 0.1, 0.01, 0.01), (5.0, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01),
                   (10, 0.6, 0.01, 0.01), (10, 0.1, -1e-12, 0.01), (10, 0.1, 0.01, 0.11), (1, 0.2, 0.0, 0.0),
                   (10, 0.1, 1e-13, 0.01), (400, -0.5, 0.1, 1e-200)):
        try:
            fastest_mode(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("fastest_mode%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 6 ---
# selection without mutation (both end states absorbing, N = 2 included), with one rate at its floor 1e-12 and
# the other 0, and with frequent mutation in both directions, against the matrix in extended precision
import numpy as np


def _crown_case():
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
        # values only (the return type, shape and dtype are checked in cases 0 and 1)
        lam, mode = fastest_mode(N, s, u, v)
        return lam, np.asarray(mode, dtype=float)

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

    for N, s, u, v, dps in ((14, 0.4, 0.0, 0.0, 100), (2, -0.3, 0.0, 0.0, 40), (25, 0.45, 1e-12, 0.0, 120),
                            (30, -0.5, 0.1, 0.1, 150)):
        got, mode = _t_call(N, s, u, v)
        lam_t, mode_t = _t_mp_mode(N, s, u, v, dps)
        assert _t_rel(got, lam_t) < 1e-8, (N, s, u, v, got, lam_t)
        _t_check_mode(mode, mode_t, _t_absorbing_mask(N, u, v), 1e-8, (N, s, u, v))


_crown_case()
