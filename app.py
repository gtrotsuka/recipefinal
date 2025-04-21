from flask import Flask, request, jsonify
import pickle
import os
from recipe2 import RecipeRanker
import time

app = Flask(__name__)

print("Loading RecipeRank")
if os.path.exists('recipe_ranker.pkl'):
    with open('recipe_ranker.pkl', 'rb') as f:
        ranker = pickle.load(f)
else:
    ranker = RecipeRanker('filtered_dataset.csv', load_saved=True, data_dir="recipe_data")
    with open('recipe_ranker.pkl', 'wb') as f:
        pickle.dump(ranker, f)
# print("RecipeRanker ready")

@app.route('/rank', methods=['POST'])
def rank():
    data = request.json
    if not data or 'user_ingredients' not in data:
        #er msg
        return jsonify({'error': 'user_ingredients is required'}), 400

    user_ingredients = data['user_ingredients']
    optional_ingredients = ['water','salt','pepper','thyme', 'parsley', 'dill', 'basil', 'cilantro','nutmeg','cinnamon', 'boiling water']

    start = time.time()
    results = ranker.rank_recipes(user_ingredients, optional_ingredients=optional_ingredients, top_n=10)
    elapsed = time.time() - start

    output = [
        {
            "name": r['title'],
            "ingredients": r['ingredients'],
            "steps": r['directions']
        }
        for r in results
    ]

    return jsonify({'recipe': output})

if __name__ == '__main__':
    #false for multipro
    app.run(debug=False)
