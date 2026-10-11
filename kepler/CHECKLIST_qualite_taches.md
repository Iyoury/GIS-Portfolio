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
