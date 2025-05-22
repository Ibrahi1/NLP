import spacy
from spacy.tokens import Token
from spacy.matcher import Matcher
from spacy import displacy
import os

# Charger le modèle français
nlp = spacy.load("fr_core_news_sm")

# --- Partie 1.1 : Annotation grammaticale ---
def analyser_texte(texte):
    doc = nlp(texte)
    print(f"{'TOKEN':<15} {'POS':<10} {'LEMMA':<15} {'DEP':<10}")
    for token in doc:
        print(f"{token.text:<15} {token.pos_:<10} {token.lemma_:<15} {token.dep_:<10}")

# Solution pour corriger "3x/jour" (ex : matcher ou tokenizer personnalisé)
def nettoyer_tokenisation(doc):
    # Exemple : on peut matcher les patterns posologiques
    matcher = Matcher(nlp.vocab)
    pattern = [{"TEXT": {"REGEX": r"\d+x/.*"}}]
    matcher.add("POSOLOGIE", [pattern])
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        print(f"Posologie détectée : {span.text}")
    return matches

# --- Partie 1.2 : Extraction symptômes et traitements ---
def extraire_symptomes(doc):
    symptomes = []
    for token in doc:
        if token.pos_ == "NOUN" and token.dep_ in ("obj", "nsubj", "obl"):
            adj = [child.text for child in token.children if child.pos_ == "ADJ"]
            symptomes.append((token.text, " ".join(adj)))
    return symptomes

def extraire_traitements(doc):
    traitements = []
    for token in doc:
        if token.lemma_ == "prescrire":
            for child in token.children:
                if child.pos_ == "NOUN" and child.dep_ in ("obj", "obl"):
                    traitements.append((token.text, child.text))
    return traitements

# --- Partie 2.1 : Extraction de relations Sujet-Verbe-Objet ---
def extraire_relations(phrase):
    doc = nlp(phrase)
    relations = []
    for token in doc:
        if token.pos_ == "VERB":
            sujet = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubj:pass")]
            obj = [w.text for w in token.rights if w.dep_ in ("obj", "obl")]
            for s in sujet:
                for o in obj:
                    relations.append((s, token.lemma_, o))
    return relations

# --- Partie 2.2 : Gestion des négations ---
def detecter_negations(doc):
    neg_relations = []
    for token in doc:
        if token.pos_ == "VERB":
            neg = [w for w in token.children if w.dep_ == "advmod" and w.lemma_ == "ne"]
            if neg or "pas" in [w.text for w in token.subtree]:
                sujet = [w.text for w in token.lefts if w.dep_ == "nsubj"]
                obj = [w.text for w in token.rights if w.dep_ in ("obj", "obl")]
                for s in sujet:
                    for o in obj:
                        neg_relations.append((s, f"NE PAS {token.lemma_}", o))
    return neg_relations


import os

def sauvegarder_dependances(phrase, filename="Devoir4_5/dependances.svg"):
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    
    doc = nlp(phrase)
    svg = displacy.render(doc, style="dep", jupyter=False)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg)
    
    print(f"✅ Graphique sauvegardé dans {filename}")


# --- Partie 3.2 : Médicaments + posologies ---
def associer_medicaments_posologie(doc):
    resultats = []
    for token in doc:
        if token.pos_ == "NOUN" and token.dep_ in ("obj", "nmod"):
            medicament = token.text
            posologie = " ".join([child.text for child in token.children if child.like_num or "/" in child.text or "mg" in child.text])
            if posologie:
                resultats.append((medicament, posologie))
    return resultats

# --- TEST : 5 Phrases ---
phrases_test = [
    "Le patient refuse l'anticoagulant malgré son AVC récent.",
    "Prescription : ibuprofène 400mg si douleur, maximum 3 comprimés/jour.",
    "Pas d'antibiothérapie pour cette infection virale.",
    "Le médecin arrête l'aspirine en raison de saignements.",
    "Après analyse, le cardiologue recommande un scanner cardiaque immédiat."
]

# --- EXECUTION TEST ---
if __name__ == "__main__":
    for phrase in phrases_test:
        doc = nlp(phrase)
        print("\n--- Phrase :", phrase)
        print("Relations :", extraire_relations(phrase))
        print("Négations :", detecter_negations(doc))
        print("Symptômes :", extraire_symptomes(doc))
        print("Traitements :", extraire_traitements(doc))
        print("Posologie :", associer_medicaments_posologie(doc))

    # Visualisation sur UNE phrase uniquement (en dehors de la boucle)
    print("\n--- Visualisation avec displacy ---")
    sauvegarder_dependances("Après analyse, le cardiologue recommande un scanner cardiaque immédiat.")