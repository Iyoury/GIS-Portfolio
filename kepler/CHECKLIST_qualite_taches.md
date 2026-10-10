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
