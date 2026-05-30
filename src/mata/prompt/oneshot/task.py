gqa_task_title = "Compositional image question answering question"

gqa_task_desc = \
"""This type of question is intended to return a textual answer to the given question. 
Please use `final_answer` as the varible name when providing Python code. Make sure `final_answer` is string type.
E.g., For the question 'What sport can you use this for?', please provide the name of the sport as your answer in the final step.
E.g., For the question 'Is it good weather?', the final answer must be either 'yes' or 'no'.
"""

refcoco_task_title = "Referring Expression Comprehension"

refcoco_task_desc = \
"""This type of task is to return one image patch in the image that corresponds best to the given query. 
The object described by the query must exist in the image, and only have one patch. You need to first detect that kind of object in the image and then identify which one matches the description in the query. 
Please use `final_answer` as the target image patch name when providing Python code. Make sure only one ImagePatch in `final_answer`.
E.g., query is 'left woman with shoes,' return one of the detected woman patches in the final step, don't return shoes patch.
E.g., query is 'muffins on the table,' return one of the muffin patches in the final step, don't return table patch.
E.g., query is 'white chaise under window',return one of the chaise patches in the final step, don't return window patch.
"""
