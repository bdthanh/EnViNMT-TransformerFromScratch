from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import set_up_necessary_objects, translate

app = FastAPI()

model, src_tokenizer, trg_tokenizer, device, config = set_up_necessary_objects()

class TranslationInput(BaseModel):
    input_sentence: str

@app.post("/translate/")
async def perform_translation(translation_input: TranslationInput):
    try:
        translated_sentence = translate(
            translation_input.input_sentence, config, model, src_tokenizer, trg_tokenizer, device
        )
        return {"translated_sentence": translated_sentence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))