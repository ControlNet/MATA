from dataclasses import dataclass


@dataclass
class CodeExample:
    query: str
    executed_code: str
    python_code: str

    def to_prompt(self) -> str:
        return f"""<Query>{self.query}</Query>
<ExecutedCode> 
{self.executed_code} 
</ExecutedCode>
Output:
<PythonCode>
```python
{self.python_code}
```
</PythonCode>
"""


def combine_examples(examples: list[CodeExample]) -> str:
    return "\n".join([example.to_prompt() for example in examples])


gqa_code_example = combine_examples([
    CodeExample(
        query="Does the tool on top of the table look clean and black?",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""# find tool
tool_patches = image_patch.find(['tool'])['tool']
# only one fool has been detected, get tool patch
tool_patch = tool_patches[0]
# Verify if the tool is clean 
tool_is_clean = tool_patch.verify_property(object_name='utensil', attribute='clean')
# verify if the tool is black
tool_is_black = tool_patch.verify_property(object_name='utensil', attribute='black')
# get final answer
final_answer = bool_to_yesno((tool_is_clean and tool_is_black))
""".strip()),
    CodeExample(
        query="Does the hat on top of the table look red?",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""# find table
table_patches = image_patch.find(['table'])['table']
# only one table has been detected, get table patch
table_patch = table_patches[0]
# get image patch above the table_patch
on_top_of_table_patch = get_patch_above_of(table_patch)
# find hat on top of the table
hat_on_top_table_patches = on_top_of_table_patch.find(['hat'])['hat']
# get hat patch
hat_patch = hat_on_top_table_patches[0]
# get the color of the hat
hat_color = hat_patch.simple_query(question='What is the color of the hat?')
# get final answer
final_answer = bool_to_yesno('red' in hat_color)
""".strip()),
    CodeExample(
        query="How big is the toy?",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""# Get the answer to a basic question 'The toy is big or small in this image?' asked about the image. Return the answer as final answer.
final_answer = image_patch.simple_query('The toy is big or small in this image?')""".strip()),
    CodeExample(
        query="What kind of device is this?",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""# find device
device_patches = image_patch.find(['device'])['device']
device_patch = device_patches[0]
device_name = device_patch.simple_query('What kind of device is this?')
final_answer = device_name""".strip()),
    CodeExample(
        query="Are there bagels or lemons?",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""# Check for lemon existence 
has_lemon = image_patch.exists("lemon")
# check for bagels existence
has_begel = image_patch.exists("bagels")
# get final answer, either has_lemon or has_begel is true return 'yes'
final_answer = bool_to_yesno(has_lemon or has_begel)""".strip()),
])

refcoco_code_example = combine_examples([
    CodeExample(
        query="man in blue holding banana",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""# find man
man_patches = image_patch.find(['man'])['man']
man_in_blue_patches = []
for man_patch in man_patches:
    if man_patch.verify_property("clothes", "blue"):
        man_in_blue_patches.append(man_patch)
final_answer = best_image_match(man_in_blue_patches, ['holding banana'])""".strip()),
    CodeExample(
        query="bowl of carrots",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""# find bowl
all_patches = image_patch.find(['bowl'])
person_patches = all_patches['bowl'] 
final_answer = best_image_match(person_patches, ['bowl of carrots'])""".strip()),
    CodeExample(
        query="middle kid",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""# find kid
kid_patches = image_patch.find(['kid'])['kid']
middle_kid_patch = get_middle_patch(kid_patches)
final_answer = middle_kid_patch""".strip()),
    CodeExample(
        query="left front person",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""# find people
people_patches = image_patch.find(['people'])['people']
# sort image patches
left_to_right_people_patches = get_sorted_patches_left_to_right(people_patches)
# only get a half of people who are on left hand size.
half_people_on_left_patches = left_to_right_people_patches[0:len(people_patches)//2]
# sort people on left from front to back and get the left front person
left_front_person = get_sorted_patches_front_to_back(half_people_on_left_patches)[0]
final_answer = left_front_person""".strip()),
    CodeExample(
        query="middle lady in orange",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""lady_patches = image_patch.find(['lady'])['lady']
lady_in_orange_patches = []
for lady_patch in lady_in_orange_patches():
    if lady_patch.verify_property("clothes", "orange"):
        lady_in_orange_patches.append(lady_patch)
final_answer = get_middle_patch(lady_in_orange_patches)""".strip()),
    CodeExample(
        query="seated player",
        executed_code="image_patch = ImagePatch(image)",
        python_code="""player_patches = image_patch.find(['player'])['player']
best_img_match_seated_player = best_image_match(player_patches, ['seated player'])
final_answer = best_img_match_seated_player""".strip()),
])
