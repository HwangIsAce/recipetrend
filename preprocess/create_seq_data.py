
if __name__ == "__main__":
    import pandas as pd

    food = pd.read_csv('/disk1/data/Food.com/RAW_recipes.csv')
    food.head()

    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline

    tokenizer = AutoTokenizer.from_pretrained("Dizex/FoodBaseBERT")
    model = AutoModelForTokenClassification.from_pretrained("Dizex/FoodBaseBERT")

    pipe = pipeline("ner", model=model, tokenizer=tokenizer)

    from tqdm import tqdm
    recipe_seq = []
    # cnt = 0
    for steps in tqdm(food['steps']):
        steps = eval(steps)
        # print(steps)
        recipe_seq_tmp2 = []
        for step in steps:
            ner_entity_results = pipe(step)
            # print(ner_entity_results)
            recipe_seq_tmp1 = []
            for v in ner_entity_results:
                # print(v['word'])
                recipe_seq_tmp1.append(v['word'])
            recipe_seq_tmp2.append(recipe_seq_tmp1)
            # pritnt('')
        recipe_seq.append(recipe_seq_tmp2)
        # cnt +=1
        # if cnt > 1 : break

    recipe_seq = [str(n) for n in recipe_seq]

    import pandas as pd
    recipe_seq_df = pd.DataFrame(recipe_seq)

    recipe_seq_df.to_csv('/home/jaesung/jaesung/research/recipetrend/preprocess/recipe_seq_1.csv')