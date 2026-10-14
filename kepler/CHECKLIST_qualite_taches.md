# Projet Kepler : checklist de qualité des tâches (bande difficile)

Reçue le 10 octobre 2026. Contexte : une tâche résolue 0 fois sur 8 essais frontière ne passe pas
automatiquement, elle part en « needs review » et c'est cette checklist que la revue applique.
À relire avant chaque livraison de bundle.

## La checklist (texte d'origine)

1. **Agents fail on the science, not the setup.** None of the 8 attempts failed from
   infrastructure, a timeout, a refusal or a format error.
2. **The hard part is discoverable.** The effect being graded shows up in the visible data or
   follows from standard domain knowledge.
3. **Tolerances are fair.** Gates are stated in the instruction and calibrated so that correct
   methods pass and plausible wrong ones fail clearly.
4. **There is independent proof.** A second solution or an independent ground truth passes, not
   just the author's reference.
5. **It isn't a guessing game.** The difficulty comes from a real scientific judgment, not from an
   unwritten rule or a repeated trap.
6. **The verifier is robust and the task is feasible.** Grading is binary and offline, and the
   reference finishes well within the time limit.

## Ce que cela impose à nos bundles

- Point 1 : tester l'image de l'environnement et celle du vérificateur sans réseau, chronométrer
  la référence, garder les formats de sortie simples (CSV, JSON) et les décrire dans l'instruction.
- Point 2 : l'effet noté doit être visible dans les données ou découler du savoir du domaine ;
  les conventions et les défauts des données sont décrits dans le README des données. Un piège
  qu'aucune donnée ne révèle est à proscrire.
- Point 3 : les portes sont nommées dans l'instruction (quels axes sont jugés) et posées dans les
  trous de la calibration six seeds : la référence et l'indépendante passent avec marge, les
  pipelines dégradés plausibles échouent nettement ; une porte « pas toujours fatale » se
  documente dans calibration.txt.
- Point 4 : l'implémentation indépendante (code séparé, mécanique différente) doit passer les six
  seeds ; vérifier aussi la clé contre une dérivation indépendante (exemple : la direction de
  pendage de la tâche carottes orientées, où la clé était fausse de 180 degrés et où les huit
  agents avaient raison).
- Point 5 : la difficulté doit venir d'un jugement scientifique (identifier un artefact, auditer
  un levé, corriger un biais d'échantillonnage), pas d'une règle non écrite ni d'un piège répété.
- Point 6 : reward binaire via reward.txt et ctrf.json, vérificateur numpy seul sans réseau,
  référence en secondes, valeurs non finies rejetées avant toute comparaison.

## Leçon du cas carottes orientées (v3 à v4)

Le 0/8 de la v3 venait de la clé, pas des agents : un cas « verifier correctness » que la revue
aurait rejeté. Une fois la clé corrigée, quatre sondes Opus locales ont passé la v4 en 21 à 29
commandes : chaque complication spécifiée est lisible par un agent qui lit tout. La marge de
difficulté honnête est dans le jugement (points 2 et 5), pas dans l'empilement de conventions.

## Leçon du cas carottes orientées (v4 à v5)

La v4, entièrement spécifiée, a été résolue 2/3 par la sonde de facilité (Opus, effort faible).
La v5 ajoute un jugement non écrit mais lisible dans les données : un trou sans levé de déviation
dont les six voisins montrent qu'il dévie, et dont la déviation se retrouve en ajustant ses
structures aux familles. Le vérificateur juge l'orientation dans chaque trou (porte par trou), de
sorte que tenir ce trou droit coûte la tâche (14-26° au 80e percentile contre 5° pour la
référence). Deux sondes Opus locales (effort par défaut) l'ont trouvé seules, dont une sans la
moindre phrase du README : le point est équitable (point 2) mais un Opus appliqué le découvre ;
la sonde de facilité à effort faible tranchera. Trois pièges de générateur corrigés au passage :
passes inversées adjacentes (une passe bonne coincée entre deux stretches est indécidable), trou
atteignant la verticale (taux non identifiables), structure à cheval sur une limite de passe
(passe de la profondeur consignée, pas de la profondeur interne). Règle retenue : quand les deux
implémentations se trompent au même endroit, c'est le générateur ou la porte qu'il faut corriger,
pas les solveurs.

## Leçon du cas carottes orientées (v5 à v6)

La v5 a passé la sonde de facilité mais la sonde de difficulté a résolu 7/7 (un essai en erreur
plateforme). Deux leviers testés : (a) le biais de Terzaghi appliqué plan par plan (orientations
échantillonnées inclinées vers l'axe du trou, famille coupée en deux lobes par la zone aveugle) :
l'effet sature à 3,5-4° à l'échelle du programme quelle que soit la géométrie, trop faible pour une
porte nette, écarté ; (b) une famille qui n'existe que dans un domaine (joints de décompression
confinés à la roche altérée, visibles dans lithology.csv et dans la distribution en profondeur,
rien d'écrit) : l'espacement compté sur toute la longueur des trous est faux d'un facteur 7 à 9,
porte nette, retenu. La sonde locale (Opus, effort par défaut) l'a quand même trouvé en 17
commandes : un jugement découvrable est découvert par un agent appliqué. Règles retenues :
chaque tolérance écrite dans l'instruction (point 3) ; quand une famille partage une région du
stéréonet avec un artefact, les proportions du mélange doivent pouvoir varier avec la profondeur ;
la condition « pas de famille miroir d'une autre autour d'un axe de trou » doit tenir tout le long
du trou, pas seulement aux extrémités (une indécidabilité à mi-trou a produit 13 fausses passes
dans l'indépendante). Au-delà, pour la bande « résolu rarement », il faut un sujet où l'agent ne
peut pas vérifier sa propre réponse par la cohérence interne des données.

## Leçon du cas dykes magnétiques (`dyke-swarm-remanence`)

Sujet construit après la consigne « il faut en sciences plus dur, pas la structure » : douze dykes
de trois essaims en éventail, relevé magnétique au sol et affleurements (susceptibilité des
échantillons, pas de paléomagnétisme), partition en essaims et pôle de chaque essaim. Les leviers
sont tous physiques : non-unicité induit/rémanent, composante le long de la direction invisible
dans un dyke 2D, direction commune résolue seulement par l'essaim, éventails qui se recouvrent,
rémanence visqueuse le long du champ actuel (reconnue par l'incohérence du modèle naïf et par le
magnétisme des roches), anomalie = contraste avec l'encaissant (susceptibilité du socle donnée).
Ce que les sondes locales ont appris :

- La première version (quatre leviers « manuel ») a été résolue en 12 commandes par un Opus à
  effort par défaut, avec une précision de référence. Un levier qui est du savoir standard est
  appliqué directement par un agent fort : il fait la faisabilité, pas la difficulté.
- La rémanence visqueuse (levier de jugement : le modèle naïf devient incohérent, il faut nommer
  la cause et estimer conjointement le rapport) a été trouvée par la seconde sonde en 18
  commandes, via le χ² du regroupement. Un jugement découvrable par incohérence est découvert
  par un agent appliqué ; la bande de difficulté se joue sur le nombre de tels jugements.
- Un essaim à deux polarités (trois dykes normaux, un inversé) a été construit et abandonné :
  avec une contrainte à une dimension par dyke, un dyke inversé isolé est indécidable entre son
  essaim et l'antipode de la projection d'un autre essaim ; le générateur ne trouvait plus
  d'instance séparable. Règle : avant d'ajouter un levier de jugement, vérifier qu'il reste
  identifiable avec les degrés de liberté des données (point 5), sinon c'est un jeu de devinettes.
- Rapport visqueux déterminé à ±0,1 près seulement par les angles : la précision des directions
  d'essaim est limitée par l'erreur de pendage d'affleurement propagée par la dégénérescence
  pendage/direction. Les portes ont été fixées sur ce que les données permettent (test avec des
  affleurements sans erreur : 1°), pas sur ce que la référence obtient par chance.
- Les deux sondes ont utilisé le module exact du champ total ; le générateur projetait l'anomalie
  sur le champ (écart de 60 nT sur les dykes forts). Règle : le générateur mesure ce que
  l'instrument mesure, pas l'approximation d'interprétation.
- Les instances sont sélectionnées pour que les données décident (déplacement ou échange d'un
  dyke à ≥ 4σ) ; c'est dit dans le README et dans les limites.

## Leçon du cas flux de chaleur (`borehole-heat-flow`)

Sujet choisi sur la règle « chaîne de réduction sans retour » tirée des neuf tâches passées :
cinq corrections standard (Horner avec le registre de forage, anisotropie et pendage, zone
d'écoulement, paléoclimat, production de chaleur), portes calibrées, indépendante à 0,12 mW/m² de
la référence. La sonde Opus locale (effort par défaut) a tout appliqué en 10 commandes. La règle
était incomplète : une correction **annoncée dans les données** (log lithologique, débitmètre,
histoire de surface fournie) est une étape de lecture pour un agent fort, même sans résidu à
minimiser. Ce que les tâches passées avaient en plus, ce sont des **artefacts d'instrument
silencieux**, nombreux, à découvrir par le contrôle qualité des enregistrements bruts (bouteille
de zéro vide, saut d'horloge, MNT biaisé, sauts de gravimètre, retours nuageux). Règle complétée :
le bon sujet est le dépouillement d'enregistrements **bruts** d'instruments, avec une dizaine de
défauts réalistes non étiquetés, chacun décalant le résultat hors porte, et des métadonnées qui
permettent de les trouver (journal de bord, certificats, dates) sans les nommer.

## Leçon de la similarité (`borehole-heat-flow`, rejeté avant la vérification)

Le bundle a été rejeté par le contrôle de similarité (« trop proche d'une tâche existante ») avant
toute épreuve. Un sujet de manuel (flux de chaleur en forage, Horner, Bullard) a toutes les chances
d'avoir déjà été pris par un autre auteur. Règle : choisir un sujet **rare** dont la chaîne de
réduction est longue par nature (un instrument, une campagne, un contractant), et le vérifier
contre la liste de ses propres tâches avant de construire ; un concept de manuel, même bien fait,
peut être perdu sur la similarité seule.

## Leçon du cas spectrométrie gamma aéroportée (`airborne-radiometric-reduction`)

Construit sur la règle complétée : la base de données brute d'un contractant (spectres de 1 s à
256 canaux, canaux auxiliaires, pads, ligne test à cinq hauteurs dans les deux sens, vols haute
altitude, passages sur l'eau), la chaîne IAEA standard, et cinq défauts silencieux que rien
n'étiquette (retard de position, dérive de gain sur quatre lignes, décrochages et pics du
radar-altimètre, bouffée de radon sous inversion, ligne avortée et revolée). Ce que la
construction a appris :

- Les **fenêtres doivent être définies en canaux** dans les métadonnées : entre « canaux
  entiers » et « énergies exactes » l'écart sur un rapport de stripping atteint 0,07, plus que la
  porte. Toute convention qui peut se lire de deux façons doit être fixée par les données.
- La vérité d'une constante **mesurée sur le terrain** doit être sa valeur apparente (ce qu'un
  calcul sans bruit donne avec la méthode du praticien), pas celle du modèle ponctuel : l'empreinte
  au sol grandit avec la hauteur sur un sol hétérogène et décale les coefficients d'atténuation de
  quelques pour cent. Sinon la référence échoue sur une porte que ses propres données ne permettent
  pas d'atteindre.
- Vérifier les **débordements de type** des données générées (spectres de pads en uint16 saturés
  à 65 535 : rapports de stripping faussés de 0,02 avant correction).
- Les portes par **moyennes ligne-unité** (≥ 40 enregistrements) laissent passer le bruit
  statistique des enregistrements de 1 s et attrapent toute erreur systématique sur quelques
  lignes ou un vol ; le bruit de la fenêtre U a imposé un détecteur de 33,6 L et un plancher de
  0,6 ppm, à calibrer sur six graines avant de fixer quoi que ce soit.
- Un estimateur indépendant qui compare l'aéroporté aux lectures au sol pour le retard est biaisé
  par la géométrie des contacts (l'empreinte voit les unités voisines) ; seule la comparaison des
  passages opposés annule cette géométrie. Une implémentation indépendante doit être différente
  dans l'estimateur, pas dans le principe physique quand il n'y en a qu'un de juste.

Ce que les trois sondes locales ont appris (version à cinq défauts, puis à huit défauts dont trois de
jugement : constante périmée datée, panne du détecteur vers le haut, spectres de pads dans l'ordre
d'acquisition) : résolues en 24, 21 et 15 commandes, la dernière à **effort bas**, tous les défauts
trouvés et corrigés, y compris en déclarant invalide une constante du dossier à cause de sa date. Règle
à retenir : avec ce niveau d'agent, « enregistrements bruts + défauts silencieux découvrables par le
contrôle qualité » ne suffit plus ; chaque défaut supplémentaire coûte deux commandes. Toujours faire une
sonde locale **à effort bas** avant de soumettre : c'est elle qui imite la sonde de facilité. La seule voie
restante est une science sans procédure de manuel (la méthode standard fausse pour ces données, la
correction à dériver), avec le risque de la revue d'équité.

## Leçon de la version relief (`airborne-radiometric-reduction` v3)

La plateforme a confirmé la sonde locale sur la v2 (facilité 2/3 à effort bas). La v3 remplace la
source de difficulté : deux gorges et une crête traversent le bloc, l'avion tient le niveau du plateau,
et l'hypothèse du sol plat de la correction de hauteur IAEA donne ×1,5 à ×3 dans les gorges ; la
correction se dérive des premiers principes (radiance isotrope, cos θ_n · e^(−μr)/r², intégrée sur le
MNT fourni), elle n'est dans aucun manuel. Sonde locale à effort bas : **échec** (tout le reste juste,
relief ignoré), première fois sur ce sujet. Règles :

- La difficulté qui tient est **une physique sans recette** que les données rendent visible (MNT fourni,
  altimètre à 250 m, concentrations ×3 dans une même unité) ; les défauts d'instrument, même nombreux,
  ne tiennent pas à ce niveau d'agent.
- Juger le levier **dans ses propres cellules** (cellules de relief, tolérances élargies pour le bruit
  des comptes) : dilué dans des moyennes ligne-unité ordinaires, un biais de ×3 sur dix enregistrements
  passait.
- Pour une correction géométrique, la vérité doit être insensible à l'erreur de position de l'agent
  (parois lisses, pentes < 35°) ; des parois raides faisaient varier le facteur de 100 % en 100 m et
  la référence elle-même échouait.
- Ordre des étapes : corriger le relief **avant** l'étalonnage sur la ligne test, sinon la ligne test
  porte le biais ; choisir la ligne test là où le facteur de terrain vaut 1.
- Le dépiquage de l'altimètre se fait contre le dégagement prédit (GPS − géoïde − MNT), pas par médiane
  glissante, qui prend les gorges pour des pics.
- `pkill -f` : jamais dans une commande qui contient aussi le nom du script visé (le shell s'est tué
  deux fois) ; motif entre crochets et commande séparée.

## Leçon de la version « détecteur haut mort » (`airborne-radiometric-reduction` v4)

Deux sondes locales de plus à effort bas sur la v3 : la deuxième a **résolu** la tâche en 15 commandes
(somme sur les cellules du MNT pour le relief, a_u re-mesuré, pads par date, interpolation des trois
lignes mortes). Un seul levier à ~50 % de réussite n'est pas assez. Mesures et règles :

- **Un seuil ne distingue pas les modèles, seulement l'absence d'une étape.** Deux corrections de
  terrain approximatives (sources ponctuelles sans cosinus de la normale ; correction de hauteur au
  dégagement moyen de l'empreinte, sans intégrale) sont à 2-3 % de la vérité dans les cellules de relief,
  la référence à 1-2 %, la terre plate à 10-50 % ; aucune tolérance juste ne sépare les premières de la
  référence (bruit de Poisson). Le dire dans le README (c'est un argument d'équité) et ne pas
  compter sur la fidélité du modèle pour la difficulté.
- **Instrument mort sur un vol entier plutôt que sur trois lignes.** Le détecteur haut meurt après le
  passage sur l'eau du vol 2 et reste mort sur ses treize lignes pendant que la bouffée de radon décroît
  (τ 1,5 h) : rien à interpoler en temps (prochain enregistrement valide le lendemain), la valeur
  maintenue est fausse de 1 ppm en fin de vol, et le radon doit venir des croisements avec les lignes de
  rattachement (vol 3, détecteur valide) — la pratique du métier, nommée nulle part. Variantes « zéros
  pris pour des données » et « dernière valeur maintenue » : échec net ; la référence et l'indépendante
  (estimateurs différents : différence par croisement + moyenne ±300 s, contre nivellement par ligne
  avec arbre k-d et quadratique en temps) passent.
- **Mais** une sonde à effort bas a fait ce nivellement par croisements en 17 commandes (sans toucher
  au relief) : les leviers « contrôle qualité » et « pratique du métier » tombent chacun ~une fois sur
  deux à effort bas ; c'est leur **produit** qui compte (deux leviers indépendants à 50 % → 25 % par
  essai, ≈ 84 % de passer la sonde de facilité à ≤ 1/3).
- **Définir l'amplitude d'un défaut dans l'unité jugée** (ppm), pas en comptes bruts : la bouffée de
  30-40 cps valait 1,4 à 4 ppm selon l'efficacité du pack, et la variante naïve passait à 95 % de
  cellules sur une graine. Mettre l'amplitude à l'échelle de la sensibilité.
- La revue de qualité exige une section **« Reference solution » dans le README** (pas seulement
  `solution_explanation` dans task.toml) et vérifie les chiffres du texte contre le code (5 vs 9
  enregistrements de la moyenne glissante). Écrire les textes **à partir du code**, constantes
  incluses, et relire chaque nombre.
- Le motif d'un kill (`grep "[a]grcal"`) ne doit pas partager sa commande avec la ligne de relance qui
  contient le nom en clair : le shell s'est tué une troisième fois.

### Complément v4 : trois sondes de plus à effort bas (Opus, 15-19 commandes chacune)

- Sonde 8 (détecteur mort sur tout le vol 2) : **résolue** — nivellement par croisements sur les rattachements
  **et** correction de relief par lancer de rayons sur le MNT, en 17 commandes.
- Troisième levier ajouté : une source de contrôle (radium) oubliée dans la soute au vol 3 (spectre additif
  constant sur toutes les fenêtres ; le détecteur haut ne la voit pas ; le passage QC sur la ligne test et le
  passage sur l'eau la montrent ; les rattachements la portent dans le nivellement du vol 2 ; a_u biaisé si le
  passage sur l'eau du vol 3 est pris tel quel). Variantes « prise pour du sol » (cellules eU 12 %) et « retirée
  comme un facteur » (74 %) : échec net ; référence (passage QC contre le vol d'étalonnage, deux itérations) et
  indépendante (passage sur l'eau) : passent.
- Sonde 9 (données livrées, trois leviers) : **échec** — mais elle a mesuré la source sur le passage sur l'eau
  (15,5 cps, vraie 15,7) et nivelé le vol 2 sur les rattachements en 19 commandes ; seul le relief l'a arrêtée.
- **Bilan sur cinq sondes à effort bas** : croisements sur rattachements 3/3, source oubliée 1/1, pads/dérive de
  gain/a_u/retard 5/5 ; relief 2/5. À ce niveau d'agent, **tout contrôle qualité « du métier » est fait
  systématiquement** (il vérifie les passages QC, l'eau, les croisements sans qu'on le lui dise) ; les défauts
  d'instrument, même en chaîne, ne retiennent rien. Le seul levier qui tient est la physique que l'agent doit
  *décider* d'appliquer (le relief), et il la décide ~40 % du temps malgré l'indice obligatoire des portes.
- Conséquence pour la suite : ne plus empiler de défauts QC ; chercher des leviers de physique dont l'oubli est
  silencieux et dont la correction n'est pas une recette (le relief en est un ; en trouver un deuxième par tâche).
- Sonde 10 (données livrées) : échec par le relief seul ; elle a **absorbé la source oubliée comme du radon** par
  croisements des rattachements avec le vol 1 (le spectre d'une source U et celui du radon ont les mêmes rapports
  de fenêtres) : un levier « contamination additive » est annulé par n'importe quel nivellement par croisements.
  Bilan final v4 sur données livrées : 0/2 à effort bas, par le relief dans les deux cas.
- Boucle a_u → radon → décalages de vol → a_u : instable (gain > 1), point fixe faux sur 4 graines. Règle :
  **une constante mesurée sur un passage de référence ne se révise pas** avec des quantités qui en dépendent ;
  prendre le passage au signal le plus fort et s'y tenir. Vérifier toujours les constantes estimées contre la
  vérité sur toutes les graines, pas seulement le reward.

## Leçon de la v5 (`airborne-radiometric-reduction`) : une constante retirée du paquet

Sonde de difficulté de la plateforme sur la v4 : **7 réussites sur 7 notées** à effort xhigh (un essai en erreur
d'infrastructure) ; la sonde de facilité est passée. Les agents forts font toute la chaîne. Levier v5 : les
incréments de hauteur des rapports de stripping ne sont plus dans le paquet (le manuel donne 0,00049/0,00065 par
mètre, le pack synthétique vaut 2 à 3 fois plus, build-up tiré par graine) ; une terrasse à thorium dominant
(eTh 45-65, eU 2,5-4,5, K ~1 %) de 2 × 1,4 km sur la ligne test, survolée à cinq hauteurs, permet de les mesurer ;
ils deviennent un livrable noté à 25 % près. Règles tirées :

- **Un paramètre ne se mesure que sur un sol qui l'isole.** L'ajustement conjoint sur toute la ligne test
  (U et Th corrélés le long de la ligne) était mal posé : valeurs négatives. Il faut un tronçon où l'élément
  contaminant domine (Th ≫ U, K) : l'IAEA le dit (calibration range), le générateur doit le fournir.
- **L'empreinte impose la taille de la cible** : une terrasse de 170-230 m nord-sud biaisait β de 50 % aux
  passages hauts (l'empreinte à 165 m a 300 m de rayon et voyait le K de l'encaissant). Largeur ≥ 2 × 700 m,
  contraste de K faible, marges de 600 m (sensibilités à 80 m) et 900 m (atténuation à 170 m) : la queue
  d'empreinte d'un sol à 50 ppm de Th contre 8 autour biaisait la sensibilité Th de 9 % avec 350 m de marge.
- **Découpler les ajustements** : atténuation et sensibilités sur le sol ordinaire de la ligne test, incréments
  sur la terrasse seule ; sinon α', μ_U et S_U forment une boucle qui converge à côté (μ_U 0,0056 pour 0,0075).
- **Les croisements de nivellement se comparent sur des taux strippés** : sur la terrasse, la fuite de Th dans la
  fenêtre U dépend de la hauteur de chaque trace, et la différence brute se lisait comme du radon (0,8 ppm sur
  une ligne entière du vol mort).
- **Porte relative, pas absolue**, quand la vraie valeur varie d'un facteur 2 entre graines : 0,0003/m valait 12 à
  30 % selon la graine ; 25 % relatif garde la discrimination (manuel à 50-75 %).
- Sonde locale à effort bas (22 commandes) : a **dérivé les incréments à 9 % près** par un ajustement conjoint
  « valeurs au sol fixées » (et a échoué par le relief seul). Même ce levier ne résiste pas à un agent compétent ;
  il ajoute une étape à réussir, pas une barrière.
- Complément v5, suite : (a) les croisements de nivellement se font aussi **hors sol à thorium anormal** et entre traces
  à dégagement voisin (±40 m), et la ligne test se garde à plus de 600 m de l'anomalie (un croisement sur l'anomalie
  se lisait comme 50 cps de radon) ; (b) la porte de la sensibilité U est liée à celle des incréments (un tiers à la
  moitié de la fenêtre U est du stripping sur la ligne test) : porte U à 15 %, K/Th à 10 % ; (c) une sonde xhigh
  locale (42 commandes) a tout résolu, incréments compris ; les deux sondes à effort bas ont dérivé les incréments
  à 7-9 % près et échoué par le relief seul. Une constante retirée du paquet **ajoute une étape, pas une barrière**.

### Radiométrie v5 — verdict de la plateforme (14 oct., 9 h 48)

Sonde de facilité **échouée : 2 réussites sur 3** à effort bas (10-15 min, 35-44 k jetons de sortie par essai),
après que la v4 avait passé cette même sonde (0/3) et échoué la sonde de difficulté (7/7 à xhigh). Lecture :
retirer une constante du paquet (incréments de stripping) n'a rien ajouté comme barrière, et le levier du
relief, qui arrêtait 2 essais sur 3 à effort bas en v3-v4, s'est trouvé franchi 2 fois sur 3 cette fois : à
effort bas la réussite tient au tirage, pas à la tâche. Le squelette « gamma aéroporté » est abandonné (il
figure d'ailleurs au catalogue des squelettes à ne plus reprendre). Règle : **quand un levier ne tient que
60 % du temps à effort bas, il ne porte pas une tâche ; on change de sujet au lieu d'empiler des artefacts.**
