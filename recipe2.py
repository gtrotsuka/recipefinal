import numpy as np
import pandas as pd
from scipy import sparse
from fast_pagerank import pagerank_power
import csv
import ast
import re
import os
import pickle  
import time
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
from functools import partial
import heapq
from collections import defaultdict

class RecipeRanker:
    def __init__(self, csv_file, load_saved=True, data_dir="data"): 
        print("Init...")
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True) 
        self.recipes = self.loadrec(csv_file)
        print(f"Loaded {len(self.recipes)} recipe")
        self.ingredient_index = None
        self.cooc_matrix = None

        if load_saved and self.checksave():
            self.load()
            print("Loaded ing")
        else:
            self.ingredient_index, self.cooc_matrix = self.build_ing(self.recipes)
            print(f"Built ingredient graph {len(self.ingredient_index)}")
            self.save() 
        self.ingredient_variants = {
        ing: self.normalize_ingredient(ing) 
        for ing in self.ingredient_index
    }

    def normalize_ingredient(self, ingredient):
        ingredient_synonyms = {
            "chicken breast": "chicken",
            "chicken breasts": "chicken",
            "chunk white chicken": "chicken",
            "cubed chicken": "chicken",
            "chicken thigh": "chicken",
            "chicken thighs": "chicken",
            "vegetable oil": "oil",
            "olive oil": "oil",
            "canola oil": "oil",
            "cooking oil": "oil",
            "margarine":"butter",
            "ground beef": "beef",
            "ground turkey": "turkey",
            "ground pork": "pork",
            "ground chicken": "chicken",
            "brown sugar": "sugar",
            "white sugar": "sugar",
            "granulated sugar": "sugar",
            "powdered sugar": "sugar",
            "cooking spray": "oil",
            "all-purpose flour": "flour",
            "self-rising flour": "flour",
            "cheddar cheese": "cheese",
            "cream cheese": "cheese",
            "parmesan cheese": "cheese",
            "parmesan": "cheese",
            "mozzarella cheese": "cheese",
            "mozzarella": "cheese",
            "blue cheese": "cheese",
            "feta cheese": "cheese",
            "goat cheese": "cheese",
            "ricotta cheese": "cheese",
            "sharp cheddar": "cheese",
            "green pepper": "bell pepper",
            "green peppers": "bell pepper",
            "vidalia onion": "onion",
            "vidalia onions": "onion",
            "yellow onion": "onion",
            "yellow onions": "onion",
            "sweet onion": "onion",
            "sweet onions": "onion",
            "red onion": "onion",
            "red onions": "onion",
            "white onion": "onion",
            "white onions": "onion",
            "buttermilk": "milk",
            "almond milk": "milk",
            "soy milk": "milk",
            "coconut milk": "milk",
            "oat milk": "milk",
            "rice milk": "milk",
            "milk substitute": "milk",
            "milk alternative": "milk",
            "milk powder": "milk",

            
        }
        ing = re.sub(r'\b(\w+ed|fresh|chopped|ground|sliced|grated)\b', '', ingredient.lower())
        ing = re.sub(r's\b', '', ing).strip()  # s's's
        normalized_name = ' '.join(ing.split())
        return ingredient_synonyms.get(normalized_name, normalized_name)

    def checksave(self):
        index_file = os.path.join(self.data_dir, 'ingredient_index.pkl')
        matrix_file = os.path.join(self.data_dir, 'cooc_matrix.npz')
        return os.path.exists(index_file) and os.path.exists(matrix_file)
    def loadrec(self, csv_file):
        # print(f"Loading recipes from {csv_file}")
        recipes = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            recipe_count = 0
            for row in reader:
                ner_str = str(row.get('NER', '')).strip()
                try:
                    ner_list = ast.literal_eval(ner_str) if ner_str else []
                except (SyntaxError, ValueError):
                    #erorr
                    ner_list = []

                recipes.append({
                    'id': row['id'],
                    'title': row['title'],
                    'ingredients': ast.literal_eval(row['ingredients']),
                    'directions': ast.literal_eval(row['directions']),
                    'ner_list': ner_list
                })
                recipe_count += 1
                if recipe_count % 100000 == 0:
                    print(f"Loaded {recipe_count} recipes so far...")
                if recipe_count >= 50000:
                    break
        print("Finished load")
        return recipes

    def build_ing(self, recipes):
        print("Building co-occurrence graph")
        ingredient_map = {ing: idx for idx, ing in enumerate(set().union(*[recipe['ner_list'] for recipe in recipes]))}
        cooc_matrix = sparse.lil_matrix((len(ingredient_map), len(ingredient_map)))
        # self.cooc_matrix = cooc_matrix.tocsr()

        total_recipes = len(recipes)
        for recipe_idx, recipe in enumerate(recipes, start=1):
            ingredients = [ingredient_map[ing] for ing in recipe['ner_list']]
            for i in range(len(ingredients)):
                for j in range(i+1, len(ingredients)):
                    cooc_matrix[ingredients[i], ingredients[j]] += 1
                    cooc_matrix[ingredients[j], ingredients[i]] += 1

            # Progress Check hehe
            if recipe_idx % 100000 == 0 or recipe_idx == total_recipes:
                print(f"Processed {recipe_idx}/{total_recipes} recipes...")

        print("Finished")
        return ingredient_map, cooc_matrix.tocsr()

    def save(self):
        # print("Saving ingredients graph")
        index_file = os.path.join(self.data_dir, 'ingredient_index.pkl')
        matrix_file = os.path.join(self.data_dir, 'cooc_matrix.npz')

        with open(index_file, 'wb') as f:
            pickle.dump(self.ingredient_index, f)
        sparse.save_npz(matrix_file, self.cooc_matrix)
        print("saved")

    def load(self):

        print("Load")
        index_file = os.path.join(self.data_dir, 'ingredient_index.pkl')
        matrix_file = os.path.join(self.data_dir, 'cooc_matrix.npz')

        with open(index_file, 'rb') as f:
            self.ingredient_index = pickle.load(f)
        self.cooc_matrix = sparse.load_npz(matrix_file)
        print("load done")

    def calccomplexity(self, recipe):
        if not hasattr(self, '_complexity_cache'):
            self._complexity_cache = {}
        
        recipe_id = recipe['id']
        if recipe_id not in self._complexity_cache:
            num_ingredients = len(recipe['ner_list'])
            num_steps = len(recipe['directions'])
            max_ingredients = 20
            max_steps = 15
            ingredient_complexity = min(num_ingredients / max_ingredients, 1)
            step_complexity = min(num_steps / max_steps, 1)
            complexity = 0.6 * ingredient_complexity + 0.4 * step_complexity
            self._complexity_cache[recipe_id] = complexity
        
        return self._complexity_cache[recipe_id]

    def calcmeal_substantiation(self, recipe):
        if not hasattr(self, '_substantiation_cache'):
            self._substantiation_cache = {}
        
        recipe_id = recipe['id']
        if recipe_id not in self._substantiation_cache:
            meal_keywords = ['protein', 'vegetable', 'carbohydrate', 'main dish', 
                           'entree', 'dinner', 'lunch', 'meat', 'fish', 'chicken', 
                           'beef', 'pork', 'pasta', 'rice', 'potato']
            
            ingredients_lower = ' '.join(ing.lower() for ing in recipe['ingredients'])
            keyword_count = sum(1 for keyword in meal_keywords if keyword in ingredients_lower)
            
            ingredient_diversity = len(set(recipe['ner_list'])) / len(recipe['ner_list']) if recipe['ner_list'] else 0
            
            ingredient_count = len(recipe['ingredients'])
            ingredient_count_factor = 1 - abs(ingredient_count - 8) / 8
            ingredient_count_factor = max(0, ingredient_count_factor)
            
            title_length_factor = min(len(recipe['title']) / 20, 1)
            
            substantiation_score = (
                keyword_count / len(meal_keywords) + 
                ingredient_diversity + 
                ingredient_count_factor + 
                title_length_factor
            ) / 4
            self._substantiation_cache[recipe_id] = min(substantiation_score, 1)
        
        return self._substantiation_cache[recipe_id]

    # def precomputer(self):

    #     self.recipe_features = []
    #     for recipe in self.recipes:
    #         ingredients = [self.ingredient_index[ing] for ing in recipe['ner_list'] 
    #                      if ing in self.ingredient_index]
    #         complexity = 1 - self.calccomplexity(recipe)
    #         substantiation = self.calcmeal_substantiation(recipe)
    #         self.recipe_features.append({
    #             'recipe': recipe,
    #             'ingredients': ingredients,
    #             'complexity': complexity,
    #             'substantiation': substantiation
    #         })

    def score(self, feature_batch, pr_scores, user_normalized, optional_ingredients=set(),top_n=1000):
        top_results = []
        complexity_weight = 0.2
        substantiation_weight = 0.5

        self.match_exceptions = {
        "butter": ["peanut butter", "almond butter", "cashew butter"], 
        }
        # Normalzie
        normalized_optional = {self.normalize_ingredient(ing) for ing in optional_ingredients}

        for feature in feature_batch:
            recipe = feature['recipe']
            ingredients = feature['ingredients']
            matched = []
            for ing in recipe['ner_list']:
                normalized_ing = self.ingredient_variants.get(ing, "")
                if normalized_ing in user_normalized:
                    if ing in self.match_exceptions:
                        if any(exc in recipe['ner_list'] for exc in self.match_exceptions[ing]):
                            continue  
                    matched.append(ing)

            missing_count = len([
                ing for ing in recipe['ner_list']
                if self.ingredient_variants.get(ing, "") not in user_normalized
                and self.ingredient_variants.get(ing, "") not in normalized_optional
            ])

            # Base
            if ingredients:
                base_score = pr_scores[ingredients].sum()
                if missing_count == 0:
                    base_score *= 3
                else:
                    base_score /= (1 + missing_count**1.5)

                combined_score = (
                    (1 - complexity_weight - substantiation_weight) * base_score +
                    complexity_weight * feature['complexity'] +
                    substantiation_weight * feature['substantiation']
                )

                entry = (combined_score, recipe['id'], recipe)

                if len(top_results) < top_n:
                    heapq.heappush(top_results, entry)
                else:
                    heapq.heappushpop(top_results, entry)

        return top_results

    def rank_recipes(self, user_ingredients, optional_ingredients=None, alpha=0.85,
                 complexity_weight=0.2, substantiation_weight=0.3, top_n=1000):
        print("parallel engage")
        start_time = time.time()


        user_normalized = {self.normalize_ingredient(ing) for ing in user_ingredients}

        optional_ingredients = optional_ingredients or []

        personalization = np.zeros(len(self.ingredient_index), dtype=np.float32)
        matched_indices = [
            idx for ing, idx in self.ingredient_index.items()
            if self.ingredient_variants.get(ing, "") in user_normalized
        ]

        if matched_indices:
            personalization[matched_indices] = 1.0 / len(matched_indices)

        # Page rank
        pr_scores = pagerank_power(
            self.cooc_matrix.tocsr().astype(np.float32),
            p=alpha,
            personalize=personalization,
            tol=1e-4,
            max_iter=30
        )

        # if not hasattr(self, 'recipe_features'):
        #     self._precompute_recipe_features()
        recipe_features = []
        for recipe in self.recipes:
            ingredients = [self.ingredient_index[ing] for ing in recipe['ner_list']
                        if ing in self.ingredient_index]
            complexity = 1 - self.calccomplexity(recipe)
            substantiation = self.calcmeal_substantiation(recipe)
            recipe_features.append({
                'recipe': recipe,
                'ingredients': ingredients,
                'complexity': complexity,
                'substantiation': substantiation
            })
        # Parall
        num_cores = cpu_count()
        # batch_size = len(self.recipe_features) // num_cores + 1
        # batches = [
        #     self.recipe_features[i:i + batch_size]
        #     for i in range(0, len(self.recipe_features), batch_size)
        # ]
        batch_size = len(recipe_features) // num_cores + 1
        batches = [recipe_features[i:i + batch_size] for i in range(0, len(recipe_features), batch_size)]

        with Pool(num_cores) as pool:
            score_func = partial(
                self.score,
                pr_scores=pr_scores,
                user_normalized=user_normalized,
                optional_ingredients=set(optional_ingredients),
                top_n=top_n
            )
            batch_results = pool.map(score_func, batches)

        top_results = []
        for batch in batch_results:
            for entry in batch:
                if len(top_results) < top_n:
                    heapq.heappush(top_results, entry)
                else:
                    heapq.heappushpop(top_results, entry)
        sorted_recipes = [recipe for (score, recipe_id, recipe) in 
                        sorted(top_results, key=lambda x: (-x[0], x[1]))]

        print(f"Ranked {len(sorted_recipes)} recipes in {time.time() - start_time:.2f} seconds")
        for i, (score, recipe_id, recipe) in enumerate(sorted(top_results, key=lambda x: (-x[0], x[1])), start=1):
            matched_count = len([ing for ing in recipe['ner_list'] if self.ingredient_variants.get(ing, "") in user_normalized])
            mismatched_count = len([ing for ing in recipe['ner_list'] if self.ingredient_variants.get(ing, "") not in user_normalized])
            print(f"Rank {i}: Recipe ID {recipe_id}, Score: {score:.4f}, Title: {recipe['title']}, Matched Ingredients: {matched_count}, Mismatched Ingredients: {mismatched_count}")
        return sorted_recipes
    
# if __name__ == '__main__':
#     start = time.time()
#     pickle_file = 'recipe_ranker.pkl'
#     if os.path.exists(pickle_file):
#         print("Loading RecipeRanker from pickle file...")
#         with open(pickle_file, 'rb') as f:
#             ranker = pickle.load(f)
#     else:
#         print("Initializing RecipeRanker...")
#         ranker = RecipeRanker('filtered_dataset.csv', load_saved=True, data_dir="recipe_data")
        
#         print("Saving RecipeRanker to pickle file...")
#         with open(pickle_file, 'wb') as f:
#             pickle.dump(ranker, f)
#             elapsed_time = time.time() - start
#             print(f"Initialization took {elapsed_time:.2f} seconds")
#             start = time.time()

#     # CHANGE HERE
#     user_ingredients = ['beef','potatoes','carrot','onion','garlic','butter', 'green pepper','bell pepper','mushroom soup','lunch meat','peas']

#     optional_ingredients = ['water','salt','pepper','thyme', 'parsley', 'dill', 'basil', 'cilantro','nutmeg','cinnamon', 'boiling water']
    
#     for optional in optional_ingredients:
#         if optional not in user_ingredients:
#             user_ingredients.append(optional)
#     results = ranker.rank_recipes(user_ingredients, optional_ingredients=optional_ingredients, top_n=10)
#     output_file = 'test.csv'
#     with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['id', 'title', 'ingredients', 'directions']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         writer.writeheader()
#         for recipe in results:
#             writer.writerow({
#                 'id': recipe['id'],
#                 'title': recipe['title'],
#                 'ingredients': recipe['ingredients'],
#                 'directions': recipe['directions']
#             })

#     print(f"Results written to {output_file}")

#     print(f"Query took {time.time() - start:.2f} seconds")

