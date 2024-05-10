import replicate
import os

from openai import AzureOpenAI


def get_gen_text(
        pre_prompt,
        prompt,
        model="mistralai/mixtral-8x7b-instruct-v0.1",
        temperature=0.3,
        top_p=0.9,
        max_length=2048,
        repetition_penalty=0.5):
    input_dict = {
        "prompt": f"{pre_prompt} {prompt} Assistant: ",
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "repetition_penalty": repetition_penalty
    }

    output = replicate.run(
        model,
        input=input_dict
    )

    response = ""
    for item in output:
        response += item

    return response


def get_text_openai(
        prompt,
        preprompt, 
        model="0125-Preview",
        temperature=0.3,
        top_p=0.9,
        max_length=2048,
        repetition_penalty=0.5
        ):

        client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        azure_deployment="bf-openai-gpt4",
        api_version="2024-02-01",     
        )

        completion = client.chat.completions.create(
        model="0125-Preview",
            messages=[
                {
                    "role": "system",
                    "content":        preprompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return completion.choices[0].message.content


def get_clusters_descriptive_text(clusters_df):
    pre_prompt = """
        A continuacion te entregare estaidsticos de una serie de clusters de usuarios, que incluyen cosas como el numero de usuarios que lo componen o el promedio en metricas claves. 
        Quiero que me escribas un parrafo breve indicando las diferencias que se observan en el comportamiento de los dos, y los principales isnights que generan estas diferencias, ojala dando valor para temas de marketing o identificacion. 
        Muy importante mantener la brevedad! No explicaciones largas ni detalladas, si no que de una a los insnights
        Trata de describir los features usando la naturaleza de su data y ojala sin mucho decimal ni detalle innecesario. Por ejemplo, usar porcentaje para binarios, notacion de horario para horas, etc. 
        Asegurate de responder en español. Utiliza negritas (formato HTML) para destacar lo principal de cada descripcion. 
        A continuacion los clusters: 
    """

    num_clusters = len(clusters_df)

    if num_clusters == 0:
        return ""

    prompt = ""

    for _, row in clusters_df.iterrows():
        for column, value in row.items():
            prompt += f"{column}: {value}\n"
        prompt += "\n"

#    output = get_gen_text(
#        prompt=prompt,
#        pre_prompt=pre_prompt
#    )

    output = get_text_openai(
        prompt=prompt,
        preprompt=pre_prompt
    )

    return output

