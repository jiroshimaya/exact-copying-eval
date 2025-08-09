import litellm
from pydantic import BaseModel, Field
class Person(BaseModel):
    """Person Information"""
    name: str = Field(..., description="name of the person.")
    age: int = Field(..., description="age of the person.")

def generate_persons() -> list[Person]:
    messages_list = []
    messages_list.append([
        {"role": "user", "content": "男性の情報を生成して"}
    ])
    messages_list.append([
        {"role": "user", "content": "女性の情報を生成して"}
    ])
    response = litellm.batch_completion(
        messages=messages_list,
        model="gpt-4.1-mini",
        response_format=Person
    )
    persons = []
    for response in response:
        content = response['choices'][0]['message']['content']
        person = Person.model_validate_json(content)
        persons.append(person)
    return persons

if __name__ == "__main__":
    person = generate_persons()
    print(person)
