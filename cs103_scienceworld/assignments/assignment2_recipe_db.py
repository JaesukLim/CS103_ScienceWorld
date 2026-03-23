from typing import Dict, List


ASSIGNMENT_2_RECIPE_DOCUMENTS: Dict[str, str] = {
    "peanut butter sandwich": "To make peanut butter sandwich, you need to mix peanut, bread.",
    "jam sandwich": "To make jam sandwich, you need to mix jam, bread.",
    "banana sandwich": "To make banana sandwich, you need to mix banana, bread.",
    "mixed nuts": "To make mixed nuts, you need to mix peanut, almond, cashew.",
    "fruit salad": "To make fruit salad, you need to mix apple, orange, banana.",
}


def get_assignment_2_recipe_documents() -> Dict[str, str]:
    return dict(ASSIGNMENT_2_RECIPE_DOCUMENTS)


def get_assignment_2_recipe_corpus() -> List[str]:
    return list(ASSIGNMENT_2_RECIPE_DOCUMENTS.values())
