gqa_task_title = "Compositional image question answering question"

gqa_task_desc = \
"""This type of question is intended to return a textual answer to the given question. 
You are required to get the answer for the question. 
E.g., When the question is 'What toy is this?,' the answer should be the name of toy. Please do not return 'yes' or 'no' in this case. 
E.g., When the question is 'Is it a good weather?', the answer should be 'yes' or 'no' in this case.
E.g., When the question is 'How tall is the table', the final answer can only be 'tall' or 'short'.
        If asking about 'How large', final answer can only be 'large' or 'small'; 
        If asking about 'how clean', final answer can only be 'clean' or 'dirty'; 
        If asking about 'how wide', final answer can only be 'wide' or 'narrow'; 
        If asking about 'how big', final answer can only be 'big' or 'small'; 
        If asking about 'how old', final answer can only be 'old' or 'young'; 
        If asking about 'how hard', final answer can only be 'hard' or 'soft';
E.g., When the question is about 'Who', final answer can from like ['women', 'woman', 'girl', 'man', 'child', 'soccer player',
                                                        'policeman', 'snowboarder', 'spectator', 'skateboarder', 'umpire',
                                                        'catcher', 'athlete', 'batter', 'people', 'boy', 'audience',
                                                        'skater', 'player', 'skier', 'gentleman', 'spectators', 'driver',
                                                        'pilot', 'crowd', 'lady', 'surfer', 'men', 'customers',
                                                        'bus driver', 'pedestrian', 'mother', 'parent', 'coach', 'jockey',
                                                        'cyclist', 'baseball players']

Pay attention please!
    - Avoid making assumptions about the object's attributes. Only utilize the provided skill to acquire visual information, rather than resorting to guesses. 
        For example, 
            'what is the color of apple', please use "Get the answer to a basic question 'What color is the apple?' asked about the apple patch."
            'What kind of device is this?', please use "specify the device name."
            'What kind of toy is this?', please use "specify the toy name."
    - Please do not use best text match when there is not any options list provided in Execution Feedback or question. 
        For example, 
            'what is the color of apple', please do not use "best text match ['red', 'green', 'blue']. Please directly use "Get the answer to a basic question 'What color is the apple?'"
            'what kind of device is this', please do not use "best text match ['keybroad', 'mouse', 'screen']. Please directly use "specify the device name"
    - Generally, you can base on the context and question to do furthur exploration and get more visual information. 
    - To identify exact name or type of an object detected in an image or to get an answer to a simple question about it, utilize Skill. e.g., "Specify the name of the animal;" "get a caption answer about 'which city is this?'";
"""

refcoco_task_title = "Referring Expression Comprehension"

refcoco_task_desc = \
"""This type of query is to return one image patch in the image that corresponds best to the given query. 
The object described by the query must exist in the image, but you need to first detect that kind of object in the image and then identify which one matches the description in the query. 
E.g., When the query is 'women left yellow boots,' you need to first find all the women, then check which of them are on the left side and wearing yellow boots, and finally return one of women patch as the target image patch.; 
E.g., When the query is 'middle person in yellow holding an umbrella,' you need to, first, find all the people in the image; second, verify which ones are holding yellow umbrellas among all the found people patches; third, determine which of these people is in the middle; and finally, return one of people patch as the target image patch.
"""
