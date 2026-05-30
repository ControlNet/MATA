gqa_instruction_setting = """1. Compositional visual reasoning task revolves around utilizing the provided perception skills to gain more detailed and concrete information about the image at each step. The instructions will be translated into Python code and executed to extract information from the image. This perceptual information will subsequently be utilized as feedback to facilitate the generation of the next step instruction.
2. The entire process of compositional visual reasoning is akin to the functioning of the human brain, encompassing language understanding (comprehending the query), visual perception, acquiring feedback, and engaging in logical reasoning.
3. At the outset (Step 1), it's crucial to note that you have no direct access to the image, meaning you possess no information about the content in image initially.
4. To provide valid instructions for answering the query, you should first use the provided perception skills to gather information from image. The perception results will be provided either in the feedback. You can also view all executed Python code before this step. Relying solely on this information ensures the certainty of object existence. For example, you cannot check the color of people's clothes if they have not been previously found or detected and reported in the current feedback.
5. Then, leverage the available information from the feedback to perform logical reasoning in order to provide the next step instruction or obtain the final result.
6. Your primary responsibility is to furnish valid instructions for the subsequent step, drawing upon factors such as the query type, query, available skills, actions taken, executed Python code, and the feedback received.
7. Image patch is a crop of an image centered around a particular object.
8. Avoid guessing. If uncertain about the image, use the skills to gather additional information.
"""

refcoco_instruction_setting = """1. Compositional visual reasoning task revolves around utilizing the provided perception skills to gain more detailed and concrete information about the image at each step. The instructions will be translated into Python code and executed to extract information from the image. This perceptual information will subsequently be utilized as feedback to facilitate the generation of the next step instruction.
2. The entire process of compositional visual reasoning is akin to the functioning of the human brain, encompassing language understanding (comprehending the query), visual perception, acquiring feedback, and engaging in logical reasoning.
3. At the outset (Step 1), it's crucial to note that you have no direct access to the image, meaning you possess no information about the content in image initially.
4. To provide valid instructions for answering the query, you should first use the provided perception skills to gather information from image. The perception results will be provided either in the feedback. You can also view all executed Python code before this step. Relying solely on this information ensures the certainty of object existence. For example, you cannot check the color of people's clothes if they have not been previously found or detected and reported in the current feedback.
5. Then, leverage the available information from the feedback to perform logical reasoning in order to provide the next step instruction or obtain the final result.
6. Your primary responsibility is to furnish valid instructions for the subsequent step, drawing upon factors such as the query type, query, available skills, actions taken, executed Python code, and the feedback received.
7. Image patch is a crop of an image centered around a particular object.
8. Avoid guessing. If uncertain about the image, use the skills to gather additional information.
"""

gqa_skills = """Skill 1: Find objects only by their names, capable of finding single or multiple objects at once. Unable to directly locate objects by name along with attributes. E.g., "find apple, kid, and muffin" or "find boots."
Skill 2: Check for object existence based on object name only, which can include digital numbers, e.g., "check apple existence in human patches" and "check number 7 existence in the athlete patches"
Skill 3: Verify if object possesses the visual property. Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead checking whether the object possesses the property. E.g., "Verify if the clothing of the human in the human patch is blue." and "Verify if the table is oval within the table patch."
Skill 4: Get the best text match based on a list of text options. The text options list should be provided by information captured. e.g., When question is "Is the color of the cup green, red, yellow or black?", the instruction can be: "get the best match text from ['red cup', 'yellow cup', 'green cup', 'black cup'] to the cup patches"
Skill 5: Specify the name of object or get the answer to a basic question asked about the image or generate a caption for image as general context, e.g., "Specify the name of this type of animals;" "get a caption answer about 'which city is this?'"; "generate a caption for image as context"
Skill 6: Compare and sort the objects by their position, e.g., 'Find the left one from object patches by sorting patches from left to right.
Skill 7: Logical reasoning. E.g., Basic logical operations, comparisons, mathematical calculations and so on.
"""

refcoco_skills = """Skill 1: Find objects only by their names, capable of finding single or multiple objects at once. Unable to directly locate objects by name along with attributes. E.g., "find apple, kid, and muffin" or "find boots."
Skill 2: Check for object existence based on object name only, which can include digital numbers, e.g., "check apple existence in human patches" and "check number 7 existence in the athlete patches"
Skill 3: Verify if object possesses the visual property. Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead checking whether the object possesses the property. E.g., "Verify if the clothing of the human in the human patch is blue." and "Verify if the table is oval within the table patch."
Skill 4: Identify the best image match containing specific content, e.g., "find the best image match for 'man holding umbrella' among all men patches."
Skill 5: Calculate the distance between the edges of two image patches, e.g., "calculate the distance between the muffin patch and the kid patch."
Skill 6: Calculate the median depth of an image crop, e.g., "calculate the depth of all muffin patches."
Skill 7: Check for overlapping between a given image patch and the current image patch, e.g., "check whether the muffin patch overlaps with the kid patch."
Skill 8: Compare and sort the objects by their position, e.g., 'Find the left one from object patches by sorting patches from left to right.
Skill 9: Logical reasoning. E.g., Basic logical operations, comparisons, mathematical calculations and so on.
"""
