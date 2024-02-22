from typing import Callable, Dict
import pandas as pd
from tqdm.auto import tqdm
from operator import itemgetter
from scripts.evaluation import extract_number

def get_answers(prompt: str, agent, questions: pd.Series):
    try:
        prompts = questions.apply(lambda x: prompt.replace('{question}', x))
    except Exception as e:
        print(e)
        return e
    return prompts.apply(lambda x: agent.invoke(x).content)

    
def get_better_prompt(best_example: Dict, validation_set_with_answers: pd.DataFrame, teacher_agent, examples_other_less_good_prompts = None) -> str:
    prompt = f"""
    You are trying to optimize the prompt of a LLM to maximize its score on a task.

    You have already tried a few prompts, with these results:
    """
    for example in examples_other_less_good_prompts:
        prompt += f"Prompt: '{example['prompt']}':\nAverage score: {example['score']}\n\n---\n"

    prompt += f"""
    The best prompt for now is: {best_example['prompt']}. It achieves score {best_example['score']}. Could you improve this prompt?

    Here are the examples of the validation set with the answers for this best prompt, to help you come up with an even better prompt:
    """
    for i, example in validation_set_with_answers.iterrows():
        prompt += f'--- Example {i}:\n'
        for feature, value in example.to_dict().items():
            prompt += f"Feature: {feature.capitalize()}: has value '{value}'.\n"

    prompt += """
    ---
    Please provide an analysis of the error cases, and suggest a possible cause.
    
    Then only at the end, based on the causes for error, come up with an improved prompt. Your improved prompt should contain the placeholder '{question}' to indicate where the question should be inserted.
    Preface your suggestion of this improved prompt with '\Improved prompt:\n', and add at the end: '\nEnd of improved prompt'.
    
    Now begin!
    """
    print('='*10+ 'Here is the new full prompt'+'='*10)
    print(prompt)
    print('='*10 + 'End new full prompt' + '='*10)
    return teacher_agent.invoke(prompt).content


def optimize_prompt(logs, prompt, validation_set, student_agent, scoring_function: Callable, teacher_agent = None, n_iter = 6):
    if not teacher_agent:
        teacher_agent = student_agent

    for _ in tqdm(range(n_iter)):
        # Score current prompt
        validation_set_with_answers = validation_set.copy()

        validation_set_with_answers['prediction'] = get_answers(prompt, student_agent, validation_set['question'])
        validation_set_with_answers['extracted_prediction'] = validation_set_with_answers['prediction'].apply(extract_number)
        validation_set_with_answers['prediction_is_correct'] = scoring_function(validation_set_with_answers['extracted_prediction'], validation_set_with_answers['true_answer'])
        display(validation_set_with_answers[['prediction', 'extracted_prediction', 'prediction_is_correct']])
        print("Current prompt:", prompt)
        print('Score:', validation_set_with_answers['prediction_is_correct'].mean())
        logs.append({'prompt': prompt, 'score': validation_set_with_answers['prediction_is_correct'].mean(), 'answers': validation_set_with_answers})

        index_best_example = max(enumerate(logs), key=(lambda x: x[1]['score']))[0]
        best_example = logs[index_best_example]

            
        # Get a better prompt!
        feedback = get_better_prompt(best_example, validation_set_with_answers, teacher_agent, examples_other_less_good_prompts=[logs[i] for i in range(len(logs)) if i != index_best_example])
        print('===========================')
        print("MODEL FEEDBACK:")
        print(feedback)
        print('END OF MODEL FEEDBACK')
        print('===========================')

        if 'Improved prompt:' in feedback:
            new_prompt = feedback.split('Improved prompt:')[-1]
        elif 'improved prompt' in feedback:
            new_prompt = feedback.split('improved prompt')[-1]
        else:
            new_prompt = best_example['prompt']
        if 'End of improved prompt' in feedback:
            new_prompt = new_prompt.split('End of improved prompt')[0]
        elif 'end of improved prompt' in feedback:
            new_prompt = new_prompt.split('end of improved prompt')[0]
        prompt = new_prompt
    return logs


