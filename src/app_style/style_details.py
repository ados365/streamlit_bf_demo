def load_background_image():
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://brainfood.cl/wp-content/uploads/2023/02/header-estrategia.svg");
    background-size: cover;
    background-position: bottom 300px right -850px;
    background-repeat: no-repeat;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 25px;
    }
    
    </style>
    '''
    return page_bg_img

