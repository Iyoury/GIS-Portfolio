import math
import numpy as np
import signal as _t_signal
import time as _t_btime

# Independent targets: the closed forms of the prism potential, attraction and gradients (Nagy, Papp and
# Benedek 2000) evaluated in 60-digit arithmetic with mpmath (no cancellation at any distance), the
# vertical attraction of the columns by mpmath tanh-sinh quadrature of the lamina term over depth at 30
# digits, and gravity data of a basin computed with that quadrature at 25 digits; all computed once and
# stored below.

_T_G = 6.6743e-11

# Time budget of one call, as stated in the prompt: the call is interrupted once it exceeds the budget
# (by one second), so a solution that is too slow fails this check instead of holding up the tests.
_t_depth = [0]


def _t_budget(fn, seconds, name):
    def wrapped(*args, **kwargs):
        if _t_depth[0]:
            return fn(*args, **kwargs)

        def _alarm(signum, frame):
            raise TimeoutError("%s did not finish within its budget of %g s per call" % (name, seconds))
        try:
            old = _t_signal.signal(_t_signal.SIGALRM, _alarm)
            _t_signal.setitimer(_t_signal.ITIMER_REAL, seconds + 1.0)
            armed = True
        except (ValueError, AttributeError, OSError):      # no SIGALRM here: measure only
            armed = False
        _t_depth[0] += 1
        start = _t_btime.perf_counter()
        try:
            out = fn(*args, **kwargs)
        finally:
            _t_depth[0] -= 1
            if armed:
                _t_signal.setitimer(_t_signal.ITIMER_REAL, 0.0)
                _t_signal.signal(_t_signal.SIGALRM, old)
        elapsed = _t_btime.perf_counter() - start
        assert elapsed <= seconds, ("%s took %.1f s (budget %g s per call)" % (name, elapsed, seconds))
        return out
    return wrapped


def _t_rel(a, b):
    return abs(a - b) / abs(b)


prism_gravity = _t_budget(prism_gravity, 10.0, "prism_gravity")

_T_BA, _T_RA = (0.0, 100.0, 0.0, 200.0, 50.0, 300.0), 2670.0
_T_BB, _T_RB = (-500.0, 500.0, -500.0, 500.0, 10.0, 12.0), -300.0


def _t_check1(p, bounds, rho, U_t, g_t):
    L = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    U, g = prism_gravity(np.array(p, float), bounds, rho)
    assert type(U) is float and isinstance(g, np.ndarray) and g.shape == (3,), (U, g)
    g_t = np.array(g_t, float)
    assert _t_rel(U, U_t) < 1e-10, (p, U, U_t)
    tol = 1e-10 * np.linalg.norm(g_t) + 1e-14 * _T_G * abs(rho) * L
    assert np.all(np.abs(g - g_t) <= tol), (p, g, g_t)


# --- test case 0: far from the prism, from 2.5 to a million diagonals (|g| down to about 1e-13 G rho L) ---
_T_FAR_A = [
    ((3419.468083841879, 6838.936167683758, -32508.840413266225), 0.00002656512508099100438145185, (-7.956606708257177922878548e-11, -1.591310732517717528903638e-10, 7.717818465485889195439492e-10)),
    ((-235919.95186213477, 236069.95186213477, 33884.993123162116), 0.000002656505517618524707088312, (5.572048743571365789144135e-12, -5.572048372101477602378368e-12, -7.960068704998571571919094e-13)),
    ((3354151.966249685, 100.0, 175.0), 0.0000002656505552584668419385961, (-7.920169326073176895809003e-14, 0.0, 0.0)),
    ((100625322.76784329, 134167130.35712439, 290471795.72317433), 2.656505553396427367444657e-9, (-2.376103074843040349073944e-18, -3.16813743312384258960305e-18, -6.859017542712777036757812e-18)),
]
_T_FAR_B = [
    ((2127.2872315242894, -1772.7393596035745, 2209.196805908432), -0.0000113203978282822984725196, (1.909356681520227682979612e-9, -1.590971545118333867048148e-9, 2.013145983032322145466087e-9)),
    ((14206.93907116247, 28413.87814232494, -137796.30899027592), -0.0000002831651936712371971604173, (2.01141193696892161500276e-13, 4.022823873982244432366691e-13, -1.951093967025741146468799e-12)),
    ((-994937.6712623609, 994937.6712623609, 142144.95303748015), -2.831662899347574710067213e-8, (-1.408661283252673854411262e-14, 1.408661283252673854411262e-14, 2.012373513334739893799637e-15)),
    ((14142149.765859503, 0.0, 11.0), -2.831662842732524363435897e-9, (2.002285996679430341792631e-16, 0.0, 0.0)),
    ((424273827.1026579, 565698436.1368773, 1224737125.2363393), -2.831662842142523867464087e-11, (6.006990141519828966335263e-21, 8.009320188693106413646486e-21, 1.734017820852274260788973e-20)),
]
for _t_p, _t_U, _t_g in _T_FAR_A:
    _t_check1(_t_p, _T_BA, _T_RA, _t_U, _t_g)
for _t_p, _t_U, _t_g in _T_FAR_B:
    _t_check1(_t_p, _T_BB, _T_RB, _t_U, _t_g)

# --- test case 1: near the prism, on its corners, edges and faces, on the line of an edge and 1e-6 m from
# it, inside (the centre included, where g = 0), and a thin plate (2 m thick, 1 km wide, negative density) ---
_T_NEAR_A = [
    ((30, 40, -20), 0.004647029595518059322531897, (0.00000330112155559446269274671, 0.000007156561567138818749072275, 0.00002439525891123995774656419)),
    ((0, 0, 50), 0.005783669719005623574377812, (0.00002251653795106050632435074, 0.00002859571941767522188348318, 0.00003010015992100261210405549)),
    ((100, 200, 300), 0.005783669719005623574377812, (-0.00002251653795106050632435074, -0.00002859571941767522188348318, -0.00003010015992100261210405549)),
    ((0, 100, 175), 0.009789818009265639248296765, (0.00007229246681069329560310504, 0.0, 0.0)),
    ((50, 0, 50), 0.006345239445497427371407195, (0.0, 0.00004038549440218272253764441, 0.00004198501189717845579818864)),
    ((50, 100, 50), 0.008102910741682739331168468, (0.0, 0.0, 0.00007193938719373315019248601)),
    ((50, 100, 175), 0.01156733943801124714875562, (0.0, 1.41947524121992205204861e-65, 0.0)),
    ((10, 20, 60), 0.007008583030247066542581962, (0.0000296636766372728669217914, 0.00002814366529157195129186314, 0.0000398785047431948763708575)),
    ((-300, 100, 50), 0.00235138220978255112132904, (0.00000578648207346190481793357, 0.0, 0.000001883518733155004744094624)),
    ((150, 260, 10), 0.003617603335459918394206756, (-0.000006997465731942261213621431, -0.000009714039470350318412456028, 0.000009141565208708669444627144)),
    ((100.0, 250.0, 50.0), 0.004592676004309462514880799, (-0.000008010283016628205930747702, -0.00001965771548277877658294516, 0.0000133568423986596575284989)),
    ((-0.001, 100.0, 175.0), 0.009789745717150190905104604, (0.00007229176408681407055792856, 0.0, 0.0)),
    ((100.000001, 250.0, 50.000001), 0.00459267600965602173943157, (-0.000008010283197882915256888178, -0.00001965771552439359214995309, 0.00001335684229195053503764236)),
    ((-1e-07, -1e-07, 175.0), 0.007502168506929950490030722, (0.00004059379330595907781941405, 0.00004966117498365167621786153, 0.0)),
    ((-30.0, 1e-06, 50.000001), 0.005154339710236875125518064, (0.00001948305750122624847434645, 0.0000169029315718887989251387, 0.0000182761549607867223982051)),
    ((554.5299621460925, -320.4416351217438, 696.3476275509622), 0.001062661835321654739556056, (-0.0000007702162805699043997693396, 0.0000006349134830410408943226999, -0.0000007811127535598808127445)),
]
_T_NEAR_B = [
    ((0, 0, 0), -0.0001384409577861416535408927, (0.0, 0.0, -0.0000002466324636248089047819133)),
    ((0, 0, 11), -0.0001410555094842286174554255, (0.0, 0.0, 0.0)),
    ((500, 500, 10), -0.00007052775474211430872771273, (0.0000002813771081863264762530925, 0.0000002813771081863264762530925, -6.284716227728133726720512e-8)),
    ((600, -700, 12), -0.00004672140929608908723850578, (3.89548179967887720167732e-8, -4.752923330954311136722139e-8, 1.082441854775910691142221e-10)),
    ((0, 0, -1), -0.0001381945517099273491895688, (0.0, 0.0, -0.0000002461796974776204503138594)),
    ((499.999, 0, 11), -0.00009629028331355342872267555, (0.0000005946779769977521971080397, 0.0, 0.0)),
]
for _t_p, _t_U, _t_g in _T_NEAR_A:
    _t_check1(_t_p, _T_BA, _T_RA, _t_U, _t_g)
for _t_p, _t_U, _t_g in _T_NEAR_B:
    _t_check1(_t_p, _T_BB, _T_RB, _t_U, _t_g)

# --- test case 2: several points at once give arrays of shapes (n,) and (n, 3) with the same accuracy ---
_t_pts = np.array([r[0] for r in _T_NEAR_A[:4]] + [r[0] for r in _T_FAR_A[:3]], float)
_t_U, _t_g = prism_gravity(_t_pts, _T_BA, _T_RA)
assert isinstance(_t_U, np.ndarray) and _t_U.shape == (7,) and isinstance(_t_g, np.ndarray) and _t_g.shape == (7, 3)
for _t_k, (_t_p, _t_Ut, _t_gt) in enumerate(_T_NEAR_A[:4] + _T_FAR_A[:3]):
    _t_gt = np.array(_t_gt, float)
    assert _t_rel(_t_U[_t_k], _t_Ut) < 1e-10
    assert np.all(np.abs(_t_g[_t_k] - _t_gt) <= 1e-10 * np.linalg.norm(_t_gt) + 1e-14 * _T_G * _T_RA * 250.0)

# --- test case 3: invalid bounds, density or points raise ValueError ---
for _t_args in (((0, 0, 0), (0, 100, 0, 200, 300, 50), 1.0), ((0, 0, 0), (0, 100, 0, 200, 50), 1.0),
                ((0, 0, 0), (0, 100, 0, 200, 50, float("inf")), 1.0), ((0, 0, 0), _T_BA, float("nan")),
                ((0, 0), _T_BA, 1.0), (np.zeros((2, 2)), _T_BA, 1.0), ((0, float("nan"), 0), _T_BA, 1.0),
                (np.zeros((0, 3)), _T_BA, 1.0)):
    try:
        prism_gravity(np.array(_t_args[0], float), _t_args[1], _t_args[2])
    except ValueError:
        pass
    else:
        raise AssertionError("prism_gravity%r must raise ValueError" % (_t_args,))
