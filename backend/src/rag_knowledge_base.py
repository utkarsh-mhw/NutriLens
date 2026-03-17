
import chromadb
from chromadb.utils import embedding_functions


# Knowledge documents
# Organised into four thematic groups that map to what the LLM explanation
# needs to justify a NOVA score: category rules, additive signals, ingredient
# signals, and combination / interaction effects.


KNOWLEDGE_DOCUMENTS = [

    # NOVA Category Definitions 

    {
        "id": "nova1_def",
        "text": (
            "NOVA Group 1 — Unprocessed or minimally processed foods. "
            "These are natural foods that have undergone only minimal processes "
            "such as drying, crushing, roasting, boiling, freezing, or pasteurisation. "
            "No substances are added. Examples include fresh fruits, vegetables, "
            "plain meat, eggs, plain milk, and unsalted nuts. "
            "A product in NOVA 1 should contain a single ingredient with no additives."
        ),
        "metadata": {"category": "nova_definition", "nova_level": 1}
    },
    {
        "id": "nova2_def",
        "text": (
            "NOVA Group 2 — Processed culinary ingredients. "
            "These are substances extracted from Group 1 foods or from nature and "
            "used in home or restaurant kitchens to prepare Group 1 foods. "
            "Examples include oils pressed from seeds, butter, lard, sugar extracted "
            "from cane or beet, flour milled from grains, and table salt. "
            "They are rarely consumed alone and are used to season, cook, or prepare Group 1 foods."
        ),
        "metadata": {"category": "nova_definition", "nova_level": 2}
    },
    {
        "id": "nova3_def",
        "text": (
            "NOVA Group 3 — Processed foods. "
            "These are relatively simple products made by adding salt, sugar, oil, or "
            "other Group 2 substances to Group 1 foods. "
            "The purpose is to increase durability or to modify or enhance sensory qualities. "
            "Examples include canned vegetables with added salt, salted nuts, cured meats, "
            "freshly baked bread with only flour, water, salt, and yeast, and most cheeses. "
            "They typically have two to five ingredients and no cosmetic or flavour additives."
        ),
        "metadata": {"category": "nova_definition", "nova_level": 3}
    },
    {
        "id": "nova4_def",
        "text": (
            "NOVA Group 4 — Ultra-processed foods. "
            "These are industrial formulations made entirely or mostly from substances "
            "extracted from foods, derived from food constituents, or synthesised in a lab. "
            "They typically contain five or more ingredients, including additives whose "
            "purpose is to make the final product palatable, appealing, or habit-forming. "
            "Common markers include flavourings, emulsifiers, stabilisers, colours, "
            "sweeteners, and humectants. Examples include soft drinks, packaged snacks, "
            "instant noodles, breakfast cereals, energy bars, and most fast food."
        ),
        "metadata": {"category": "nova_definition", "nova_level": 4}
    },

    #Additive Signals

    {
        "id": "emulsifiers_signal",
        "text": (
            "Emulsifiers are a strong signal of ultra-processing (NOVA 4). "
            "Common emulsifiers found in ultra-processed foods include soy lecithin (E322), "
            "sunflower lecithin (E322), mono- and diglycerides of fatty acids (E471), "
            "polysorbate 80 (E433), carrageenan (E407), xanthan gum (E415), "
            "and guar gum (E412). "
            "Their presence indicates that the product needed industrial stabilisation "
            "to prevent separation, which is not necessary in home cooking. "
            "A product listing two or more emulsifiers is almost certainly NOVA 4."
        ),
        "metadata": {"category": "additive_signal", "additive_type": "emulsifier"}
    },
    {
        "id": "preservatives_signal",
        "text": (
            "Preservatives extend shelf life and typically indicate processed (NOVA 3) "
            "or ultra-processed (NOVA 4) classification depending on context. "
            "Sodium benzoate (E211), potassium sorbate (E202), sodium nitrite (E250), "
            "BHA (E320), BHT (E321), and EDTA are common preservatives in packaged foods. "
            "When preservatives appear alongside flavourings, colours, or emulsifiers, "
            "the product is almost certainly NOVA 4. "
            "A single preservative such as salt or vinegar in a simple product may still "
            "be NOVA 3 if no other additives are present."
        ),
        "metadata": {"category": "additive_signal", "additive_type": "preservative"}
    },
    {
        "id": "artificial_sweeteners_signal",
        "text": (
            "Artificial sweeteners are exclusive markers of ultra-processed foods (NOVA 4). "
            "Sweeteners including aspartame (E951), sucralose (E955), acesulfame-K (E950), "
            "saccharin (E954), stevia glycosides (E960), and sorbitol (E420) do not appear "
            "in unprocessed or minimally processed foods. "
            "Their presence immediately indicates industrial formulation. "
            "Sugar-free or 'light' products that use sweetener blends are consistently NOVA 4 "
            "because the sweetener blend itself is a formulation strategy, not a culinary one."
        ),
        "metadata": {"category": "additive_signal", "additive_type": "sweetener"}
    },
    {
        "id": "flavour_enhancers_signal",
        "text": (
            "Flavour enhancers are a reliable indicator of ultra-processing (NOVA 4). "
            "Monosodium glutamate (MSG, E621), disodium inosinate (E631), disodium guanylate (E627), "
            "and yeast extract are used to boost palatability beyond what natural ingredients provide. "
            "Products relying on flavour enhancers are engineered for hyper-palatability, "
            "a defining characteristic of NOVA 4. "
            "The presence of 'natural flavour', 'artificial flavour', or 'flavouring' without "
            "a specific source also indicates industrial flavour engineering."
        ),
        "metadata": {"category": "additive_signal", "additive_type": "flavour_enhancer"}
    },
    {
        "id": "colours_signal",
        "text": (
            "Synthetic or extracted colours are a marker of ultra-processing (NOVA 4) "
            "because they serve a purely cosmetic function. "
            "Common synthetic colours include tartrazine (E102), sunset yellow (E110), "
            "carmoisine (E122), allura red (E129), brilliant blue (E133), and caramel colour (E150). "
            "Natural colours such as beetroot extract or beta-carotene used in industrial products "
            "also indicate ultra-processing, as they replace the natural colour lost during "
            "intensive processing. "
            "Any product listing a colour additive is at minimum NOVA 3, and almost always NOVA 4."
        ),
        "metadata": {"category": "additive_signal", "additive_type": "colour"}
    },
    {
        "id": "modified_starches_signal",
        "text": (
            "Modified starches (E1400–E1450) are used as thickeners or texturisers in "
            "ultra-processed foods. Unlike plain starch (which is a Group 2 culinary ingredient), "
            "chemically or physically modified starches are exclusively industrial ingredients. "
            "They appear in sauces, soups, dairy desserts, and processed meat products. "
            "Modified starch in an ingredient list strongly suggests NOVA 3 or NOVA 4 classification "
            "depending on the overall additive count."
        ),
        "metadata": {"category": "additive_signal", "additive_type": "thickener"}
    },

    # Ingredient-Level Processing Signals

    {
        "id": "refined_grains_signal",
        "text": (
            "Refined grains and flours are a processing signal that contributes to NOVA 3 or 4. "
            "Ingredients such as enriched wheat flour, bleached flour, refined corn starch, "
            "and white rice flour have had fibre, germ, and micronutrients removed. "
            "Enriched flour, where vitamins are added back industrially, indicates a "
            "processing step beyond simple milling. "
            "Products built primarily on refined grain bases combined with added sugars "
            "and fats often score NOVA 4 due to the overall formulation pattern."
        ),
        "metadata": {"category": "ingredient_signal", "ingredient_type": "grain"}
    },
    {
        "id": "protein_isolates_signal",
        "text": (
            "Protein isolates and hydrolysates are exclusively industrial ingredients and "
            "strong markers of ultra-processing (NOVA 4). "
            "Ingredients such as soy protein isolate, whey protein concentrate, casein, "
            "hydrolysed vegetable protein, and textured vegetable protein are produced through "
            "industrial extraction and are never used in home cooking. "
            "Their presence signals that the product's protein content is engineered, not natural."
        ),
        "metadata": {"category": "ingredient_signal", "ingredient_type": "protein"}
    },
    {
        "id": "added_sugars_signal",
        "text": (
            "Multiple forms of added sugar in a single product strongly indicate ultra-processing. "
            "Ultra-processed products commonly use sugar disguised under many names: "
            "high-fructose corn syrup, dextrose, maltose, invert sugar, treacle, agave nectar, "
            "fruit juice concentrate, and glucose-fructose syrup. "
            "Listing several sugar variants is a formulation strategy to place individual sugars "
            "lower on the ingredient list while maintaining high overall sugar content. "
            "Products with three or more sugar variants are almost always NOVA 4."
        ),
        "metadata": {"category": "ingredient_signal", "ingredient_type": "sugar"}
    },
    {
        "id": "hydrogenated_fats_signal",
        "text": (
            "Hydrogenated or interesterified fats are industrial fats found exclusively in "
            "ultra-processed foods (NOVA 4). "
            "Partially hydrogenated vegetable oil, fully hydrogenated palm kernel oil, and "
            "interesterified soybean oil are produced through chemical processes not replicable "
            "in home cooking. "
            "Their use improves texture and shelf life but is associated with poor nutritional outcomes. "
            "Any product containing hydrogenated fat is NOVA 4 by definition under NOVA classification."
        ),
        "metadata": {"category": "ingredient_signal", "ingredient_type": "fat"}
    },

    # Combination / Interaction Effects

    {
        "id": "additive_count_rule",
        "text": (
            "A practical rule of thumb for NOVA classification: "
            "products with zero additives are typically NOVA 1 or 2. "
            "Products with one to two additives that are salt, sugar, or simple acids "
            "(like citric acid or vinegar) are typically NOVA 3. "
            "Products with three or more additives, especially from different functional categories "
            "(e.g. one emulsifier, one sweetener, one flavouring), are almost always NOVA 4. "
            "The combination of additives from multiple functional classes is the clearest "
            "indicator of industrial formulation rather than traditional food preparation."
        ),
        "metadata": {"category": "combination_rule", "rule_type": "additive_count"}
    },
    {
        "id": "refined_plus_additive_rule",
        "text": (
            "When a product combines refined ingredients (white flour, sugar, vegetable oil) "
            "with cosmetic or functional additives (emulsifiers, flavourings, colours), "
            "it is almost always NOVA 4. "
            "This is because refined base ingredients strip the natural structure of food, "
            "and additives are then required to restore texture, taste, and appearance. "
            "This pattern — deconstruct then reconstruct using additives — is the hallmark "
            "of ultra-processed food formulation and distinguishes NOVA 4 from NOVA 3."
        ),
        "metadata": {"category": "combination_rule", "rule_type": "refined_plus_additive"}
    },
    {
        "id": "short_ingredient_list_rule",
        "text": (
            "A short ingredient list (five or fewer ingredients) with recognisable whole-food "
            "components is a positive signal for lower NOVA scores (1–3). "
            "For example, a product listing only 'oats, water, salt' is NOVA 3 at most. "
            "Conversely, a long ingredient list with many technical or chemical-sounding names "
            "almost always indicates NOVA 4. "
            "Length alone is not definitive — a product with ten whole-food spices is still "
            "lower NOVA than a product with five ingredients that include two additives."
        ),
        "metadata": {"category": "combination_rule", "rule_type": "ingredient_list_length"}
    },
    {
        "id": "nutritional_markers_rule",
        "text": (
            "Nutritional composition provides supporting evidence for NOVA scores. "
            "High sugar (>15g per 100g) combined with low fibre (<2g per 100g) and high sodium "
            "(>600mg per 100g) is a nutritional fingerprint common in NOVA 4 snacks and cereals. "
            "High saturated fat (>10g per 100g) with low protein in a grain-based product "
            "suggests added industrial fats. "
            "Very low calorie-density products claiming health benefits but listing multiple "
            "stabilisers and sweeteners are typically NOVA 4 regardless of their nutritional profile."
        ),
        "metadata": {"category": "combination_rule", "rule_type": "nutritional_markers"}
    },
]

# ChromaDB setuup

_client = None
_collection = None

COLLECTION_NAME = "nutrilens_knowledge_base"


def _get_collection():

    global _client, _collection

    if _collection is not None:
        return _collection


    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    _client = chromadb.Client()  # ephemeral in-memory client
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Seed only if the collection is empty
    if _collection.count() == 0:
        _collection.add(
            ids=[doc["id"] for doc in KNOWLEDGE_DOCUMENTS],
            documents=[doc["text"] for doc in KNOWLEDGE_DOCUMENTS],
            metadatas=[doc["metadata"] for doc in KNOWLEDGE_DOCUMENTS],
        )

    return _collection


def retrieve_context(query: str, n_results: int = 3) -> str:

    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents"],
    )
    passages = results["documents"][0]  # list of strings for query index 0
    return "\n\n".join(f"- {p}" for p in passages)