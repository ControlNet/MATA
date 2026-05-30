from dataclasses import dataclass


@dataclass
class AnswererExample:
    query: str
    guess_list: list[str]
    answer: str

    def to_prompt(self) -> str:
        return f"""<Query>{self.query}</Query>
<GuessList>{self.guess_list}</GuessList>
<Answer>{self.answer}</Answer>
"""


def combine_examples(examples: list[AnswererExample]) -> str:
    prompts = []
    for i, example in enumerate(examples):
        prompts.append(f"""<Example_{i+1}>
{example.to_prompt()}</Example_{i+1}>
""")
    return "\n".join(prompts)


answerer_examples = combine_examples([
    AnswererExample(
        query="What kind of flowers are these?",
        guess_list=[
            "these flowers are purple, so lavender, lilac, iris, and hyacinth", "purple flowers"],
        answer="lilac"
    ),
    AnswererExample(
        query="What do these people on the bikes normally write and give out?",
        guess_list=["the people on bikes are police, so Tickets", "tickets"],
        answer="tickets"
    ),
    AnswererExample(
        query="What kind of cold meet is this?",
        guess_list=["what kind of meat is this is beef, so roast beef", "beef"],
        answer="beef"
    ),
    AnswererExample(
        query="Can you guess the place shown in this picture?",
        guess_list=[
            "the place is tourist attraction, so the Eiffel Tower in Paris, France", "big ben"],
        answer="big ben"
    ),
    AnswererExample(
        query="When was this type of vehicle with two equal sized wheels invented?",
        guess_list=["the vehicle is a bicycle, so 19th century", "1819"],
        answer="1800s"
    ),
    AnswererExample(
        query="What is the flavor of the pink topping on this dessert?",
        guess_list=[
            "the topping is whipped cream, so strawberry, vanilla, chocolate, and raspberry", "strawberry"],
        answer="strawberry"
    ),
    AnswererExample(
        query="How are these festive lights held in place?",
        guess_list=[
            "these festive lights are christmas lights, so with hooks clips", "string"],
        answer="string"
    ),
    AnswererExample(
        query="Who is famous for allegedly doing this in a lightning storm?",
        guess_list=[
            "what is being done is flying a kite, so Benjamin Franklin", "Charles Manson"],
        answer="Benjamin Franklin"
    ),
    AnswererExample(
        query="What is the object atop the skier's head used for?",
        guess_list=[
            "the object atop the skier's head is helmet, so protection from head injuries", "sunglasses"],
        answer="protection"
    ),
    AnswererExample(
        query="What rank is the man on the right?",
        guess_list=[
            "who is the man on the right is sailor, so seaman", "captain"],
        answer="captain"
    ),
    AnswererExample(
        query="Chemically what kind of water is in the picture?",
        guess_list=[
            "the water in the picture is waves, so salt water", "salt water"],
        answer="salt"
    ),
    AnswererExample(
        query="Is the material tweed or canvas?",
        guess_list=["the material is fabric, so fabric", "canvas"],
        answer="canvas"
    ),
    AnswererExample(
        query="Which type of meat are in the photo?",
        guess_list=["the meat in the photo is sausage, so pork", "hot dogs"],
        answer="hotdogs"
    ),
    AnswererExample(
        query="What sort of predator might there be in an area like this?",
        guess_list=[
            "this area is mountains, so predators like wolves fox", "shark"],
        answer="shark"
    ),
    AnswererExample(
        query="Can you name a sport this person could be a part of?",
        guess_list=[
            "this person is a racer, so racing such as auto", "motorcycle racing"],
        answer="racing"
    ),
    AnswererExample(
        query="Who makes the yellow top worn in this photograph?",
        guess_list=["the top is red, so brand is unknown", "Burton"],
        answer="Burton"
    ),
    AnswererExample(
        query="Is the athlete right or left handed?",
        guess_list=[
            "what is the athlete doing is playing baseball, so unclear", "right handed"],
        answer="right handed"
    ),
    AnswererExample(
        query="Is this food high or low on fat?",
        guess_list=[
            "what kind of food is this is sandwich, so depends on ingredients", "high"],
        answer="high"
    ),
    AnswererExample(
        query="Which objects shown are typically associated with small children?",
        guess_list=[
            "what objects are shown are stuffed animals, so toys", "teddy bears"],
        answer="teddy bears"
    ),
    AnswererExample(
        query="What small appliance is that stuffed animal inside?",
        guess_list=[
            "the stuffed animal is a teddy bear, so vacuum cleaner", "microwave"],
        answer="microwave"
    ),
    AnswererExample(
        query="What is this made with?",
        guess_list=["what is this is muffin, so flour sugar eggs", "oats"],
        answer="flour"
    )
])
