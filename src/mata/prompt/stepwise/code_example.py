from dataclasses import dataclass


@dataclass
class MultiStepCodeExample:
    query: str
    # steps are started from 1
    instruction_per_step: dict[int, str]
    executed_code_per_step: dict[int, str]
    execution_results_per_step: dict[int, str]
    variables_info_per_step: dict[int, str]

    def __post_init__(self):
        self.executed_code_per_step[0] = "image_patch = ImagePatch(image)"
        self.execution_results_per_step[0] = ""
        self.variables_info_per_step[0] = "image_patch: ImagePatch(0, 0, 640, 480), patch name: original_image"
        self.max_step = max(self.executed_code_per_step.keys())

    def get_previous_instructions(self, step: int) -> str:
        return "\n".join([self.instruction_per_step[i] for i in range(1, step)]).strip("\n")

    def get_current_instruction(self, step: int) -> str:
        return self.instruction_per_step[step].strip("\n")

    def get_executed_code(self, step: int) -> str:
        return "\n".join([self.executed_code_per_step[i] for i in range(0, step)]).strip("\n")

    def get_execution_results(self, step: int) -> str:
        return "\n".join([self.execution_results_per_step[i] for i in range(0, step)]).strip("\n")

    def get_variables_info(self, step: int) -> str:
        return "\n".join([self.variables_info_per_step[i] for i in range(0, step)]).strip("\n")

    def get_output_python_code(self, step: int) -> str:
        return self.executed_code_per_step[step].strip("\n")

    def to_prompt(self, step: int) -> str:
        return f"""<Query>{self.query}</Query>
<Step>{step}</Step>
<PreviousInstructions>
{self.get_previous_instructions(step)}
</PreviousInstructions>
<Instruction>
{self.get_current_instruction(step)}
</Instruction>
<ExecutedCode>
{self.get_executed_code(step)} 
</ExecutedCode>
<ExecutionResults>
{self.get_execution_results(step)}
</ExecutionResults>
<Variables>
{self.get_variables_info(step)}
</Variables>

Output:
<PythonCode>
```python
{self.get_output_python_code(step)}
```
</PythonCode>
"""

    def to_prompt_all(self) -> list[str]:
        return [self.to_prompt(step) for step in range(1, self.max_step + 1)]


def combine_examples(examples: list[MultiStepCodeExample]) -> str:
    prompts = []
    current_index = 1
    for example in examples:
        for prompt in example.to_prompt_all():
            prompts.append(
                f"<Example_{current_index}>\n{prompt}\n</Example_{current_index}>")
            current_index += 1

    return "\n".join(prompts)


gqa_code_example = combine_examples([
    # Example 1: Does the tool on top of the table look clean and black?
    MultiStepCodeExample(
        query="Does the tool on top of the table look clean and black?",
        instruction_per_step={
            1: "Find the tool in the image.",
            2: "Verify if the tool is clean and verify if the tool is black.",
            3: "Return the answer based on the verification results."
        },
        executed_code_per_step={
            1: """# Find tool in the image
tool_patches = image_patch.find(['tool'])['tool']
# Only one tool has been detected
tool_patch = tool_patches[0]""",

            2: """# Verify if the tool is clean 
tool_is_clean = tool_patch.verify_property(object_name='utensil', attribute='clean')
# Verify if the tool is black
tool_is_black = tool_patch.verify_property(object_name='utensil', attribute='black')""",

            3: """# Get final answer based on both conditions
final_answer = bool_to_yesno((tool_is_clean and tool_is_black))"""
        },
        execution_results_per_step={
            1: "Detection result: 1 tool has been detected. tool1 bounding box is [10,20,30,40].",
            2: "The verification of clean utensil in tool_patch is: True; The verification of black utensil in tool_patch is: False",
            3: ""
        },
        variables_info_per_step={
            1: "tool_patches: [ImagePatch(10, 20, 30, 40)]\ntool_patch: ImagePatch(10, 20, 30, 40)",
            2: "tool_is_clean: True\ntool_is_black: False",
            3: "final_answer: 'no'"
        }
    ),

    # Example 2: Are there bagels or lemons?
    MultiStepCodeExample(
        query="Are there bagels or lemons?",
        instruction_per_step={
            1: "Check for the existence of lemons in the image.",
            2: "Check for the existence of bagels in the image.",
            3: "Return the answer based on the existence check results."
        },
        executed_code_per_step={
            1: """# Check for lemon existence 
has_lemon = image_patch.exists("lemon")""",

            2: """# Check for bagels existence
has_bagel = image_patch.exists("bagels")""",

            3: """# Get final answer, either has_lemon or has_bagel is true return 'yes'
final_answer = bool_to_yesno(has_lemon or has_bagel)"""
        },
        execution_results_per_step={
            1: "The existence of lemon in image patch image_patch is: True",
            2: "The existence of bagels in image patch image_patch is: False",
            3: ""
        },
        variables_info_per_step={
            1: "has_lemon: True",
            2: "has_bagel: False",
            3: "final_answer: 'yes'"
        }
    ),

    # Example 3: How big is the toy?
    MultiStepCodeExample(
        query="How big is the toy?",
        instruction_per_step={
            1: "Find the toy in the image.",
            2: "Get the answer to a basic question about the toy's size and return it as the final answer."
        },
        executed_code_per_step={
            1: """# Find toy in the image
toy_patches = image_patch.find(['toy'])['toy']
# Only one toy has been detected
toy_patch = toy_patches[0]""",

            2: """# Get the answer to a basic question about the toy's size
final_answer = toy_patch.simple_query('The toy is big or small in this image?')"""
        },
        execution_results_per_step={
            1: "Detection result: 1 toy has been detected. toy1 bounding box is [50,60,70,80].",
            2: ""
        },
        variables_info_per_step={
            1: "toy_patches: [ImagePatch(50, 60, 70, 80)]\ntoy_patch: ImagePatch(50, 60, 70, 80)",
            2: "final_answer: 'small'"
        }
    )
])

# RefCOCO Examples
refcoco_code_example = combine_examples([
    # Example 1: Man in blue holding banana
    MultiStepCodeExample(
        query="man in blue holding banana",
        instruction_per_step={
            1: "Find men in the image.",
            2: "Filter for men wearing blue clothing.",
            3: "From the men in blue, find the one holding a banana and return as final answer."
        },
        executed_code_per_step={
            1: """# Find men in the image
man_patches = image_patch.find(['man'])['man']""",

            2: """# Filter for men wearing blue clothes
man_in_blue_patches = []
for man_patch in man_patches:
    if man_patch.verify_property("clothes", "blue"):
        man_in_blue_patches.append(man_patch)""",

            3: """# Find the man in blue holding a banana
final_answer = best_image_match(man_in_blue_patches, ['holding banana'])"""
        },
        execution_results_per_step={
            1: "Detection result: 3 men have been detected. man_1 bounding box is [10,20,30,40]; man_2 bounding box is [50,60,70,80]; man_3 bounding box is [90,100,110,120].",
            2: "The verification of clothes blue in man_1 is: True; The verification of clothes blue in man_2 is: False; The verification of clothes blue in man_3 is: True",
            3: ""
        },
        variables_info_per_step={
            1: "man_patches: [ImagePatch(10, 20, 30, 40), ImagePatch(50, 60, 70, 80), ImagePatch(90, 100, 110, 120)]",
            2: "man_in_blue_patches: [ImagePatch(10, 20, 30, 40), ImagePatch(90, 100, 110, 120)]",
            3: "final_answer: ImagePatch(90, 100, 110, 120)"
        }
    ),

    # Example 2: Middle kid
    MultiStepCodeExample(
        query="middle kid",
        instruction_per_step={
            1: "Find kids in the image.",
            2: "Get the middle kid and return as final answer."
        },
        executed_code_per_step={
            1: """# Find kids in the image
kid_patches = image_patch.find(['kid'])['kid']""",

            2: """# Get the middle kid patch
middle_kid_patch = get_middle_patch(kid_patches)
final_answer = middle_kid_patch"""
        },
        execution_results_per_step={
            1: "Detection result: 3 kids have been detected. kid_1 bounding box is [5,15,25,35]; kid_2 bounding box is [45,55,65,75]; kid_3 bounding box is [85,95,105,115].",
            2: ""
        },
        variables_info_per_step={
            1: "kid_patches: [ImagePatch(5, 15, 25, 35), ImagePatch(45, 55, 65, 75), ImagePatch(85, 95, 105, 115)]",
            2: "middle_kid_patch: ImagePatch(45, 55, 65, 75)\nfinal_answer: ImagePatch(45, 55, 65, 75)"
        }
    ),

    # Example 3: Left front person
    MultiStepCodeExample(
        query="left front person",
        instruction_per_step={
            1: "Find people in the image.",
            2: "Sort people from left to right and get those on the left.",
            3: "From the left people, find the front person and return as final answer."
        },
        executed_code_per_step={
            1: """# Find people in the image
people_patches = image_patch.find(['people'])['people']""",

            2: """# Sort image patches from left to right
left_to_right_people_patches = get_sorted_patches_left_to_right(people_patches)
# Only get people who are on left hand side
half_people_on_left_patches = left_to_right_people_patches[0:len(people_patches)//2]""",

            3: """# Sort people on left from front to back and get the left front person
left_front_person = get_sorted_patches_front_to_back(half_people_on_left_patches)[0]
final_answer = left_front_person"""
        },
        execution_results_per_step={
            1: "Detection result: 4 people have been detected. people_1 bounding box is [10,20,30,40]; people_2 bounding box is [50,60,70,80]; people_3 bounding box is [90,100,110,120]; people_4 bounding box is [130,140,150,160].",
            2: "The patches list has been sorted from left to right (horizontal). Now, the first patch in the list corresponds to the leftest position, while the last one corresponds to the rightest position",
            3: ""
        },
        variables_info_per_step={
            1: "people_patches: [ImagePatch(10, 20, 30, 40), ImagePatch(50, 60, 70, 80), ImagePatch(90, 100, 110, 120), ImagePatch(130, 140, 150, 160)]",
            2: "left_to_right_people_patches: [ImagePatch(10, 20, 30, 40), ImagePatch(50, 60, 70, 80), ImagePatch(90, 100, 110, 120), ImagePatch(130, 140, 150, 160)]\nhalf_people_on_left_patches: [ImagePatch(10, 20, 30, 40), ImagePatch(50, 60, 70, 80)]",
            3: "left_front_person: ImagePatch(10, 20, 30, 40)\nfinal_answer: ImagePatch(10, 20, 30, 40)"
        }
    )
])
