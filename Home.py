import streamlit as st
from streamlit_extras.app_logo import add_logo
from src.app_style.style_details import load_background_image
import os
# ===========================================================================================
# Design settings
# ===========================================================================================

path = r"app\style.css"
assert os.path.isfile(path)
with open( path ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

page_bg_img = load_background_image()
st.markdown(page_bg_img, unsafe_allow_html=True)

# add bf logo
# add_logo("./images/bf_logo_small.png")
# ===========================================================================================
# End of design settings
# ===========================================================================================

st.title('< Plataforma de Customer analytics')

st.markdown("""
Lorem ipsum dolor sit amet, ad laoreet vulputate vix, nec ei copiosae praesent. Te sit erant elitr facilis. Mei at mazim convenire signiferumque, ut vim fabulas epicuri fastidii. Sit an vide fabulas volumus.

Dolorum posidonium ad vis, vel suas fuisset id, lorem iracundia at quo. Mea cu nemore tibique praesent. Alia facilisis cu vis, mutat argumentum et qui, vix dolor graece inermis ne. Odio suas praesent ne sea.

Mei no malis bonorum minimum, et aeque tation per. His maiorum interpretaris at. Et qui animal assentior suscipiantur. Cu vel omnes detraxit omittantur, nemore erroribus intellegat sea ei. Eam tota dissentias in. Eum choro quidam admodum ne, ad etiam erant suscipiantur mel, est error alterum officiis ei.

Fugit dissentiet mei at. Mel eu posse essent percipit, quo eu iuvaret atomorum interpretaris, no pro justo fugit liber. Elit putent eripuit pro te, omnis eruditi torquatos ad per, an sed quidam senserit. Ad omnis porro veniam has, te virtute efficiantur qui. Laoreet appetere eam ea, sea et utamur nonumes epicurei, choro temporibus mei et.

Ex magna saepe ius, est labitur facilis albucius te, omnium suavitate usu cu. Vis tamquam vituperata ea. At nec habeo mundi timeam. Ridens evertitur prodesset no nam.
""")
