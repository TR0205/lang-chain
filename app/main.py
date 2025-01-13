from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

@app.get("/")
def read_root():
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[
            {"role": "user", "content": "write a haiku about ai"}
        ]
    )
    print(completion.choices[0].message)
    # return {"Hello": "Worldaa"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str = None):
#     return {"item_id": item_id, "q": q}

# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}

# @app.delete("/items/{item_id}")
# def delete_item(item_id):
#     return {"delete item"}
