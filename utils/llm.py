import os
import time

from dotenv import load_dotenv
from google import genai


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


if not API_KEY:

    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=API_KEY
)


# =========================================================
# GENERATE AI RESPONSE
# =========================================================

def generate_response(prompt, max_retries=3):
    """
    Generate a Gemini response with graceful
    handling for temporary API failures and
    quota exhaustion.
    """

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )


            # =================================================
            # CHECK RESPONSE
            # =================================================

            if (
                not response
                or not response.text
            ):

                return (
                    "AI interpretation is temporarily "
                    "unavailable. The analytical result "
                    "above was calculated directly from "
                    "the uploaded dataset."
                )


            return response.text


        except Exception as error:

            error_text = str(error)


            # =================================================
            # QUOTA EXHAUSTED — 429
            # =================================================

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            ):

                return (
                    "AI interpretation is temporarily "
                    "unavailable because the Gemini API "
                    "quota has been reached. The analytical "
                    "result above was calculated directly "
                    "from the uploaded dataset."
                )


            # =================================================
            # TEMPORARY SERVER ERROR — 503
            # =================================================

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            ):

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    time.sleep(
                        wait_time
                    )

                    continue


                return (
                    "AI interpretation is temporarily "
                    "unavailable due to a temporary "
                    "Gemini service issue. The analytical "
                    "result above was calculated directly "
                    "from the uploaded dataset."
                )


            # =================================================
            # OTHER ERRORS
            # =================================================

            return (
                "AI interpretation could not be generated. "
                "The analytical result above was calculated "
                "directly from the uploaded dataset."
            )


    # =========================================================
    # FINAL FALLBACK
    # =========================================================

    return (
        "AI interpretation is temporarily unavailable. "
        "The analytical result above was calculated directly "
        "from the uploaded dataset."
    )