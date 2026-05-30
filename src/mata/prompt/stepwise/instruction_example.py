gqa_instruction_example = """E.g.,
    When the question is about visual question answer and the question is 'Is it a good weather?'
    All Previously Taken Instruction: 
        Generate a caption for image as general context
        Get the answer to a basic question 'is it a sunny day' asked about the image
    Execution Feedback: 
        The caption is: a man is skiing.
        The answer for the image in response to the question 'is it a sunny day' is: no.
    Output: return 'no' as final answer becasue it is not a sunny day.

E.g.,
    When the question is 'Are there black bagels or green lemons?'
    All Previously Taken Instruction: 
        Generate a caption for image as general context
        Check for lemon existence and check for bagels existence
    Execution Feedback: 
        The caption is: there is a lemon and bagel.
        The existence of green lemons in image patch is: True'; The existence of black bagels in image patch is: False'
    Output: Return 'yes' as final result because green lemons are existed.

E.g.,
    When the question is 'Does the tool on top of the table look clean and black?'
    All Previously Taken Instruction:
        find tool
        Verify if the tool is clean and verify if the tool is black
    Execution Feedback:
        Detection result: 1 tool has been detected.
        The verification of clean tool in tool patch is: True; The verification of black tool in tool patch is: False
    Output: Return 'no' as final result because tool's color is not black.

E.g.,
    When the question is 'Does the hat on top of the table look red?'
    All Previously Taken Instruction:
        find hat
    Execution Feedback: 
        Detection result: 2 hat have been detected.
    Output: find the hat on top of the table and verify if this hat is red.

E.g.,
    When the question is 'Does the hat on top of the table look red?'
    All Previously Taken Instruction:
        find hat
        find the hat on top of the table
    Execution Feedback: 
        Detection result: 2 hat have been detected.
        1 hat is on top of the table
        The verification of the hat on top of the table is red in top_table_hat_patch is: True;
    Output: Return 'yes' as final answer because the hat on top of the table is red. 

E.g.,
    When the question is 'How big is the toy?'
    All Previously Taken Instruction: 
        find toy
    Execution Feedback: 
        Detection result: 1 toy has been detected.
    Output: Get the answer to a basic question 'The toy is big or small in this image?' asked about the image. Return the answer as final answer.

E.g.,
    When the question is 'What kind of device is on top of the desk?'
    All Previously Taken Instruction: 
        find device
        find the device on top of the table
    Execution Feedback: 
        Detection result: 2 devices have been detected.
        1 device is on top of the table
    Output: specify the name of device on top of the table. Return this device name as final answer

E.g.,
    When the question is 'What color is the toy?'
    All Previously Taken Instruction: 
        find toy
    Execution Feedback: 
        Detection result: 1 toy has been detected.
    Output: Get the answer to a basic question 'What color is the toy?' asked about the toy patch. Return the answer as final answer.
"""

refcoco_instruction_example = """E.g.,
    When the query is about visual grounding detection query and the query is 'left woman in blue'.
    Step 1: find woman. 
        Feedback (current state): Detection result: 3 women have been detected. woman_1 bounding box is [1,2,3,4];woman_2 bounding box is [3,4,5,6];woman_3 bounding box is [5,6,7,8];
    Step 2: Sort the women patches based on from left to right and get the leftest patch as final answer. 
        Feedback (current state): Done

E.g.,
    When the query is 'middle kid'.
    Step 1: find kid. 
        Feedback (current state): Detection result: 3 kids have been detected. kid_1 bounding box is [1,2,3,4];kid_2 bounding box is [3,4,5,6];kid_3 bounding box is [5,6,7,8];
    Step 2: Get the middle kid patch and return as final answer. 
        Feedback (current state): Done

E.g.,
    When the query is 'girl in white next to man in left'.
    Step 1: find girl and find the girls in white by verifying if girl in white clothing. 
        Feedback (current state): Detection result: 3 grils have been detected. The verification of gril in white clothing in gril_1 is: True; The verification of gril in white clothing in gril_2 is: True; The verification of gril in white clothing in gril_3 is: False
    Step 2: find men and find the man in left
        Feedback (current state): 1 man has been detected.
    Step 3: find the girl in white which is closest to the man in left among all gril_in_white patch and return as final answer
        Feedback (current state): Done

E.g.,
    When the query is 'back'.
    Step 1: find people.
        Feedback (current state): Detection result: 2 people have been detected. people_1 bounding box is [1,2,3,4];people_2 bounding box is [3,4,5,6];
    Step 2: Get the person in back and return as final answer. 
        Feedback (current state): Done

E.g.,
    When the query is 'number 17'.
    Step 1: find people.
        Feedback (current state): Detection result: 2 people have been detected. people_1 bounding box is [1,2,3,4];people_2 bounding box is [3,4,5,6];
    Step 2: find the people with number 17 by check for '17' existence among all people patch. 
        Feedback (current state): Done
"""
