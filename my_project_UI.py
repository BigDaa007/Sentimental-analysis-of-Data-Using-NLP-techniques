#Importing Libraries
import pickle
import pandas as pd
import numpy as np
import webbrowser
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input , Output , State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

#Declaring Global Variables
project_name = None
app  = dash.Dash(__name__,external_stylesheets=[dbc.themes.CYBORG],
                 meta_tags=[{'name': 'viewport',
                             'content':'width=device-width,initial-scale=1.0'}]
                 )

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "success",
}
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Etsy.com", className="display-4"),
        html.Hr(),
        html.P(
            "Content", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
                dbc.NavLink("Page 3", href="/page-3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)


#Declaring my FUNCTIONS
def load_model():
    global scrappedReviews
    global pickle_model
    global vocab
    global df
    
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    df = pd.read_csv('estypredictedreviews.csv')
    
    file = open("pickle_model.pkl",'rb')
    pickle_model = pickle.load(file)
    
    file = open("feature.pkl",'rb')
    vocab = pickle.load(file)

    
def create_UI(): 
    main_layout = html.Div([
       dcc.Location(id="url"),
       sidebar,
       content
       ])
    return main_layout


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
        if pathname == "/":
             return[
                html.H1('Sentimental Analysis via Pie Chart',
                        className='text-center font-italic font-weight-bold mt-4'),
                dcc.Graph(id='pie',figure=px.pie(data_frame=df,names=df['predictedvalue'],labels={'positive','negative'},color_discrete_sequence=px.colors.sequential.Agsunset)),
                html.Div(id = 'the_graph')
       
                   ]
        elif pathname == "/page-1":
             return[
                 html.H1('Word Cloud',
                        className='text-center font-italic font-weight-bold p-xl-5'),
                 #html.H2(id='heading4',children='Word Cloud', className='text-center display-3 mb-4', style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '30px', 'color': 'black'}),
                 html.Div([
                        dbc.Button("ALL Words",
                         id="allbt",
                         outline=True,
                         color="info", 
                         className="mr-1",
                         n_clicks_timestamp=0,
                         style={'padding':'10px','padding-right':'15px'}
                         ),
                        dbc.Button("Positve Words",
                        id="posbt",
                         outline=True,
                         color="success", 
                         className="mr-1",
                         n_clicks_timestamp=0,
                         style={'padding':'10px','padding-right':'15px'}
                         ),
                        dbc.Button("Negative Words",
                        id="negbt",
                        outline=True, 
                        color="danger",
                        className="mr-1",
                        n_clicks_timestamp=0,
                        style={'padding':'10px','padding-right':'15px'}
                        )
                        ],style={'padding-left':'15px','textAlign': 'center'}
                        ),
                        html.Div(id='container',style={'padding':'15px', 'textAlign': 'center'})
                    ]
     
             '''
                html.H1('Sentimental Analysis via Bar Graph',
                        className='text-center font-italic font-weight-bold mt-4'),
               dcc.Graph(id='bar',figure=px.bar(data_frame=df,x_axis=['predictedvalue'],y_axis =['positive','negative'])),
               html.Div(id = 'the_graph_bar') 
               '''
                
        elif pathname == "/page-2":
             return[
                html.H1('Analysis Of Comments', id = 'heading',
                        className='text-center font-italic font-weight-bold p-xl-5 display-3'),
                        dcc.Textarea(
                        id = 'textarea_review',
                        placeholder = 'Enter the review here.....',
                        style = {'width':'100%', 'height':200, 'textAlign': 'left'}),
                       html.Button(id='button_review',children='Find Review',n_clicks=0 , className='btn btn-primary btn-lg btn-block'),
                       html.Div(id = 'result',className='text-center font-italic font-weight-bold p-xl-5')
                    ]
        elif pathname == "/page-3":
           return [
                html.H1('Drop Down Menu',className='text-center font-italic font-weight-bold p-xl-5'),
                dbc.Container([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews.reviews],
                    value = scrappedReviews.reviews[0],
                    style = {'margin-bottom': '30px'}
                    
                )
                       ],
                        style = {'padding-left': '50px', 'padding-right': '50px'}
                        ),
                #html.Button(id='button_dropdown',children='Find Review',n_clicks=0 , className='btn-block btn-primary mr-1'),
                html.Div(id = 'result1',className='text-center font-italic font-weight-bold m-10')
                ]
    # If the user tries to reach a different page, return a 404 message
        return dbc.Jumbotron(
           [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
          ]
          )
    
    
@app.callback(
    Output('the_graph', 'figure'),
    [
    Input('pie', 'value')
    ]
    )
def update_graph():
    #dff=df
    piechart= px.pie(df['predictedvalue'],hole=0.3,labels={'positive','negative'})
    return piechart


@app.callback(
    Output('container','children'),
    [
        Input('allbt','n_clicks_timestamp'),
        Input('posbt','n_clicks_timestamp'),
        Input('negbt','n_clicks_timestamp'),
    ]
)
def wordcloudbutton(allbt,posbt,negbt):

    if int(allbt) > int(posbt) and int(allbt)>int(negbt):
        return html.Div([
            html.Img(src=app.get_asset_url('all.jpg'))])
    elif int(posbt) > int(allbt) and int(posbt)>int(negbt):
        return html.Div([
            html.Img(src=app.get_asset_url('positive.jpg'))
            ])
    elif int(negbt) > int(allbt) and int(negbt) > int(posbt):
       return html.Div([
           html.Img(src=app.get_asset_url('negative.jpg'))
           ])
    else:
        pass
    

'''
@app.callback(
    Output('the_graph_bar', 'figure'),
    [
    Input('x_axis', 'value'),
    Input('y_axis', 'value')
    ]
    )

def update_bar_graph(x_axis,y_axis) :
    
    barchart= px.bar(data_frame = df,y=y_axis,x=x_axis)
    return barchart
'''
@app.callback(
    Output('result', 'children'),
    [
    Input('button_review', 'n_clicks')
    ],
    [
    State('textarea_review', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
@app.callback(
    Output('result1', 'children'),
    [
    Input('dropdown', 'value')
    ]
    )
def update_dropdown(value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
    
def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def check_review(reviewText):

    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))


    # Add code to test the sentiment of using both the model
    # 0 == negative   1 == positive
    
    return pickle_model.predict(vectorised_review)




#Main function 
def main():
    global project_name
    global scrappedReviews
    global app
    print("Start of my Project")
    load_model()
    open_browser()
    project_name="Sentimental Analysis"
    #print("My project name is : ",project_name)
    #print("My scrapped data is : ",scrappedReviews.sample(5))

    app.title= project_name
    app.layout = create_UI()
    app.run_server()
    
    
    
    print("End of my project")
    project_name = None
    scrappedReviews= None
    app=None
#calling Main function
if __name__ == '__main__':
       main()
