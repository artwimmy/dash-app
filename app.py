#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import statsmodels.formula.api as smf
#from pandas_datareader import wb
import pandas_datareader as web
import plotly.express as px
#from datetime import datetime
#from jupyter_dash import JupyterDash
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

#from dash_bootstrap_templates import load_figure_template
#from pandas_datareader import wb
from datetime import datetime, timedelta
import plotly.graph_objects as go
#from dash import dash_table as dt
#from dash.exceptions import PreventUpdate
#from dash import dcc
from dash import dash_table as dt
import dash
import math

#import dash_table_experiments as dt
#from datetime import datetime, timedelta
#from dash import dash_table as dt
#import dash_table_experiments as dt

plt.style.use('ggplot')



test = pd.read_csv(r'Final.csv')
#test = pd.read_csv(r'C:\Users\Jonas Hasle\OneDrive\Skrivebord\Final.csv')

pd.options.mode.chained_assignment = None  # default='warn'

#test = frame[~frame['Sector'].isin(['Healthcare', 'Energy'])]
#test['size score'] = test.groupby('Sector')['Market Cap'].rank(ascending=True)

test['revenue_growth_score'] = test['Revenue Growth Quarter'].dropna().astype(float).rank(method= "average",ascending=True, pct=True).copy()*100
test['gross_margin_score'] = test['Gross Margin Quarter'].astype(float).rank(method= "average", ascending=True, pct=True).copy()*100
test['assets_turnover_score'] = test['Asset Turnover Quarter'].astype(float).rank(method= "average",ascending=True, pct=True).copy()*100
test['operating_margin_score'] = test['Operating Margin Quarter'].astype(float).rank(method= "average",ascending=True, pct=True).copy()*100
test['size_score_score'] = test['Market Cap'].rank(method= "average",ascending=True, pct=True)*100
test['rev_score'] = test.groupby('Sector')['Revenue Growth Year'].rank(method= "average",ascending=True, pct=True)*100
test['mar_score'] = test.groupby('Sector')['Gross Margin Year'].rank(method= "average",ascending=True, pct=True)*100
test['style_score'] = test['rev_score'] + test['mar_score']
test['five_year_growth'] = test['5 Year Sales Growth 5YA'].dropna().astype(float).rank(method= "average",ascending=True, pct=True).copy()*100
test['five_year_margin'] = test['Gross Margin 5YA'].dropna().astype(float).rank(method= "average",ascending=True, pct=True).copy()*100

#'Price to Cash Flow MRQ'
#'Price to Book MRQ'
test['equity_growth_score'] = test['Equity Growth Quarter'].rank(method= "average",ascending=True, pct=True).copy()*100
#test['style_score'] = test['StyleScore'].rank(ascending=True)
test['pbratio_score'] = test['Price to Book MRQ'].astype(float).rank(method= "average",ascending=True, pct=True).copy()*100
test['psratio_score'] = test['Price to Sales TTM'].astype(float).rank(method= "average",ascending=True, pct=True).copy()*100
test['pcf_score'] = test['Price to Cash Flow MRQ'].astype(float).rank(method= "average",ascending=True, pct=True).copy()*100
test['SalesOneYear_score'] = test['Revenue Growth Year'].astype(float).rank(method= "average",ascending=True, pct=True).copy()*100
test['peratio_score'] = test['P/E Ratio TTM'].astype(float).astype(float).rank(method= "average",ascending=True, pct=True).copy()*100
#test['grossmargin_avg_score'] = test['GrossMarginAvg'].rank(ascending=True)
#test['revenue_growth3_score'] = test['Revenue_Growth3'].rank(ascending=True)




test['combined_alpha'] = ((test['revenue_growth_score'])*145+
                         (test['gross_margin_score'])*30 + 
                         (test['assets_turnover_score'])*80 + 
                          (test['operating_margin_score'])*26+
                         (test['SalesOneYear_score'])*10+
                          (test['size_score_score'])*110+
                         # (test['growth_score'])*10 +
                          (test['style_score'])*10+
                         (test['five_year_growth'])*3+
                         (test['equity_growth_score'])*9+
                         (test['pbratio_score'])*4+
                         (test['psratio_score'])*4+
                         (test['pcf_score'])*4 +
                         (test['peratio_score'])*4+
                         (test['five_year_margin'])*15)
                         #(test['revenue_growth_score3'])*3
                         #)


test = test.sort_values('combined_alpha', ascending=False)


# In[654]:


df = test
df['Next Earnings'] = pd.to_datetime(df['Next Earnings'])
df.rename(columns = {'index':'Company'}, inplace = True)

for i in range(len(df.columns)):
    if df[df.columns[i]].dtype == df[df.columns[1]].dtype:
            df[df.columns[i]] = df[df.columns[i]].round(2)
    else:
        pass



df = df[['Company', 'Sector','Dividend Growth Rate ANN',
'Dividend Yield ANN',
'Return on Equity TTM',
'Return on Assets TTM',
'Price to Sales TTM',
'Price to Cash Flow MRQ',
'P/E Ratio TTM',
'Price to Book MRQ',
'5 Year Sales Growth 5YA',
'Gross Margin 5YA',
'Ticker',
'Revenue Growth Quarter',
'Asset Turnover Quarter',
'Gross Margin Quarter',
'Operating Margin Quarter',
'Market Cap',
'Equity Growth Quarter']]

#frame['YTD'] = frame['YTD'].astype(str).str.replace("--","").str.replace("","").astype(str).str.replace("%", "").str.replace(",", "").replace("-", np.nan)
df.replace([np.inf, -np.inf], 0, inplace=True)
df = df.dropna()
millnames = ['',' Thousand',' Mn',' Bn',' Tn']

def millify(n):
    #n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(np.floor(0 if n == 0 else np.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

new_col = [0]*len(df)
for i in range(len(df['Market Cap'])):
               new_col[i] = millify(df['Market Cap'][i])
df.insert(loc=2, column="Market Capitalization", value=new_col)

tic = df['Ticker']
df = df.drop(columns=['Ticker'])
df.insert(loc=0, column='Ticker', value=tic)

options= [{'label': x, 'value': x}
          for x in df.columns[4:]]

datepicker = dcc.DatePickerRange(
    id = 'my_dates',
    min_date_allowed = datetime(2015, 1, 1),   # allow user to go back as far as January the 1st, 2015
    max_date_allowed = datetime.today(),       # user cannot select future dates
    start_date = datetime(2021, 1, 1),
    end_date = datetime.today()
)
factors = dcc.Dropdown(
    id = 'factors',                
    options = options, 
    value = options[0]['value'],   
    multi = True,
    clearable = True,
    searchable=True,
    placeholder="Select Factors"
)

description = """
An application that gives you the optimal companies to buy based on your investing strategy. Select your desired factors for the companies you wish to invest in and 
weight the factors according to your preference. The stocks are then ranked providing you the top stocks to buy according to your inputs.
"""

card1 = dbc.Card(children=[html.Br(),
                           dbc.Row(
                               dbc.Col(html.H1("Strategy Finder",
                                               # className='text-center text-primary mb-4',
                                               style={'textAlign': 'center', 'font-style': ['bold'], 'color': 'black'}),
                                       width=12)),

                           #     html.H1('Strategy Finder', style={'textAlign':'center','font-style': ['bold']}), #style={'textAlign': 'center','backgroundColor':'#E23744','color': 'white','font-family':['Open Sans','sans-serif'],
                           # 'font-style': ['italic'],'padding-top':'20px','padding-bottom':'20px','fontSize':17}),
                           # html.Label('Select the weights you wish to assign to each factor'),
                           dcc.Markdown(description),
                           html.Br(),
                           factors,
                           # dcc.Interval(max_intervals=1, id='inter'),

                           # dbc.Table.from_dataframe(df),
                           # dbc.Row(html.Div([html.Label('Revenue Growth MRQ', id='revenue_growth')]), style = {'display': 'block'}),
                           dbc.Row(html.Div(id='slider_rev_growth_wrapper', children=[
                               html.Label('Revenue Growth MRQ'),
                               dcc.Slider(id='revenue_growth',
                                          min=0,
                                          max=200,
                                          step=1,
                                          marks=None,
                                          # vertical=True,
                                          # handleLabel ={"label": "Weight", "showCurrentValue" : True},
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_asset_turnover_wrapper', children=[
                               html.Label('Assets Turnover MRQ'),
                               dcc.Slider(id='asset_turnover',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          # size=50,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_gross_margin_wrapper', children=[
                               html.Label('Gross Margin MRQ'),
                               dcc.Slider(id='gross_margin',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          # size=100,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_operating_margin_wrapper', children=[
                               html.Label('Operating Margin MRQ'),
                               dcc.Slider(id='operating_margin',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_market_cap_wrapper', children=[
                               html.Label('Market Capitalization'),
                               dcc.Slider(id='market_cap',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_equity_growth_wrapper', children=[
                               html.Label('Equity Growth MRQ'),
                               dcc.Slider(id='equity_growth',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_div_growth_wrapper', children=[
                               html.Label('Dividend Growth Rate ANN'),
                               dcc.Slider(id='div_growth',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_div_yield_wrapper', children=[
                               html.Label('Dividend Yield ANN'),
                               dcc.Slider(id='div_yield',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_ROE_wrapper', children=[
                               html.Label('Return on Equity TTM'),
                               dcc.Slider(id='ROE',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_ROA_wrapper', children=[
                               html.Label('Return on Assets TTM'),
                               dcc.Slider(id='ROA',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_PS_wrapper', children=[
                               html.Label('Price to Sales TTM'),
                               dcc.Slider(id='PS',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_PCF_wrapper', children=[
                               html.Label('Price to Cash Flow MRQ'),
                               dcc.Slider(id='PCF',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_five_sales_wrapper', children=[
                               html.Label('5 Year Sales Growth 5YA'),
                               dcc.Slider(id='fivesales',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_five_marg_wrapper', children=[
                               html.Label('Gross Margin 5YA'),
                               dcc.Slider(id='five_marg',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_PE_wrapper', children=[
                               html.Label('P/E Ratio TTM'),
                               dcc.Slider(id='PE',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Row(html.Div(id='slider_PB_wrapper', children=[
                               html.Label('Price to Book MRQ'),
                               dcc.Slider(id='PB',
                                          min=0,
                                          max=200,
                                          marks=None,
                                          step=1,
                                          value=0,
                                          tooltip={"placement": "bottom", "always_visible": True})],
                                            style={'display': 'block'})),

                           dbc.Container(id='my_output'),
                           dcc.Store(id='my_output2'),
                           dcc.Store(id='my_output3')
                           ],
                 className='container')

# style={"width": "18rem"})
# card2 = dbc.Card(html.Div(id='table_2'))

# html.Div(id="table_2", children=[html.Br(),
#     dbc.Label('Resulting Stocks',style={'textAlign':'center'})],
#     #dbc.Container(id = 'table_2')],
#             className = 'container'))

card3 = dbc.Card(children=[html.Br(),
                           datepicker,

                           html.H2('Resulting Stocks Performance', style={'textAlign': 'center'}),
                           dbc.Row(children=[dcc.Graph(id='price_plot'), dcc.Graph(id='sector_pie')])],
                 className='container')


#from dateutil.relativedelta import relativedelta
#import datetime
import json

theme = dbc.themes.BOOTSTRAP
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[theme],
           meta_tags=[{'name' :'viewport',
                       'content':'width=device-width, initial-scale=0.5'}])
server = app.server

app.title = 'ELA Strategy Selector'
#dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
app.layout = dbc.Container(
    children=[html.Img(src=app.get_asset_url('newimage.png')), card1, card3],
    className='dbc',
    # style = {
    #    'textAlign' : 'left',
    #    'backgroundColor' : 'lightblue',
    #    'color' : 'black'
    # }
    style={'textAlign': 'left', 'backgroundColor': 'lightblue', 'color': 'black',
           'font-family': ['Open Sans', 'sans-serif'],
           'font-style': ['bold'], 'padding-top': '20px', 'padding-bottom': '40px', 'fontSize': 19, "width": "200"},
    fluid=True
)


@app.callback(
    Output('slider_rev_growth_wrapper', 'style'),
    [Input('factors', 'value')])
def update_rev_growth(factor):
    if 'Revenue Growth Quarter' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_asset_turnover_wrapper', 'style'),
    [Input('factors', 'value')])
def update_asset_turnover(factor):
    if 'Asset Turnover Quarter' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_gross_margin_wrapper', 'style'),
    [Input('factors', 'value')])
def update_gross_margin(factor):
    if 'Gross Margin Quarter' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_operating_margin_wrapper', 'style'),
    [Input('factors', 'value')])
def update_operating_margin(factor):
    if 'Operating Margin Quarter' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_market_cap_wrapper', 'style'),
    [Input('factors', 'value')])
def update_market_cap(factor):
    if 'Market Cap' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_equity_growth_wrapper', 'style'),
    [Input('factors', 'value')])
def update_equity_growth(factor):
    if 'Equity Growth Quarter' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_div_yield_wrapper', 'style'),
    [Input('factors', 'value')])
def update_div_yield(factor):
    if 'Dividend Yield ANN' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_div_growth_wrapper', 'style'),
    [Input('factors', 'value')])
def update_div_growth(factor):
    if 'Dividend Growth Rate ANN' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_ROE_wrapper', 'style'),
    [Input('factors', 'value')])
def update_roe(factor):
    if 'Return on Equity TTM' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_ROA_wrapper', 'style'),
    [Input('factors', 'value')])
def update_roa(factor):
    if 'Return on Assets TTM' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_PS_wrapper', 'style'),
    [Input('factors', 'value')])
def update_ps(factor):
    if 'Price to Sales TTM' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_PCF_wrapper', 'style'),
    [Input('factors', 'value')])
def update_pcf(factor):
    if 'Price to Cash Flow MRQ' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_five_sales_wrapper', 'style'),
    [Input('factors', 'value')])
def update_five_sales(factor):
    if '5 Year Sales Growth 5YA' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_five_marg_wrapper', 'style'),
    [Input('factors', 'value')])
def update_five_marg(factor):
    if 'Gross Margin 5YA' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_PE_wrapper', 'style'),
    [Input('factors', 'value')])
def update_pe(factor):
    if 'P/E Ratio TTM' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('slider_PB_wrapper', 'style'),
    [Input('factors', 'value')])
def update_pb(factor):
    if 'Price to Book MRQ' in factor:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('my_output', "children"),
    Output('my_output2', "data"),
    Output('my_output3', "data"),
    [Input('factors', 'value'),
     Input('asset_turnover', 'value'),
     Input('revenue_growth', 'value'),
     Input('gross_margin', 'value'),
     Input('operating_margin', 'value'),
     Input('market_cap', 'value'),
     Input('div_growth', 'value'),
     Input('div_yield', 'value'),
     Input('ROE', 'value'),
     Input('ROA', 'value'),
     Input('PS', 'value'),
     Input('PCF', 'value'),
     Input('fivesales', 'value'),
     Input('five_marg', 'value'),
     Input('PE', 'value'),
     Input('PB', 'value'),
     Input('equity_growth', 'value')])
# Input('inter', 'n_intervals')])

def update_frame(factors, assetturns, revenueg, grossmarg, opmarg, market, divg, divy, roe, roa, ps, pcf, fiverev,
                 fivemar, pe, pb, eqgr, df=df):
    cols = []
    # factors = ['Revenue Growth Quarter', 'Asset Turnover Quarter']
    cols = []
    if type(factors) != list:
        factors = [factors]
        cols = ['Ticker'] + ['Company'] + ['Sector'] + ['Market Capitalization'] + factors
    else:
        comp = ['Ticker'] + ['Company'] + ['Sector'] + ['Market Capitalization']
        cols = list(np.append(comp, factors))
    # else:
    #   cols=[]

    filtered_df = df[cols]

    filtered_df['Combined Score'] = 0
    if type(factors) == list:
        for i in range(len(factors)):
            try:
                filtered_df[str(factors[i] + " Score")] = round(
                    filtered_df[factors[i]].astype(float).rank(method="average", ascending=True, pct=True).copy() * 100,
                    2)
            except:
                pass
    else:
        for i in range(len([factors])):
            try:
                filtered_df[str(factors[i] + " Score")] = round(
                    filtered_df[factors[i]].astype(float).rank(method="average", ascending=True, pct=True).copy() * 100,
                    2)
            except:
                pass

    filtered_df = pd.DataFrame(filtered_df)

    filtered_df['Combined Score'] = 0
    if "Revenue Growth Quarter" in factors:
        filtered_df['Combined Score'] = round(filtered_df["Revenue Growth Quarter Score"] * revenueg, 3)
    else:
        next
    if "Asset Turnover Quarter" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Asset Turnover Quarter Score"] * assetturns, 2)
    else:
        next
    if "Gross Margin Quarter" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Gross Margin Quarter Score"] * grossmarg, 2)
    else:
        next
    if "Operating Margin Quarter" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Operating Margin Quarter Score"] * opmarg, 2)
    else:
        next
    if "Market Cap" in factors:
        filtered_df['Combined Score'] = round(filtered_df['Combined Score'] + filtered_df["Market Cap Score"] * market,
                                              2)
    else:
        next
    if "Equity Growth Quarter" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Equity Growth Quarter Score"] * eqgr, 2)
    else:
        next

    if "Dividend Growth Rate ANN" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Dividend Growth Rate ANN Score"] * divg, 2)
    else:
        next
    if "Dividend Yield ANN" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Dividend Yield ANN Score"] * divy, 2)
    else:
        next
    if "Return on Equity TTM" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Return on Equity TTM Score"] * roe, 2)
    else:
        next
    if "Return on Assets TTM" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Return on Assets TTM Score"] * roa, 2)
    else:
        next
    if "Price to Sales TTM" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Price to Sales TTM Score"] * ps, 2)
    else:
        next
    if "Price to Cash Flow MRQ" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Price to Cash Flow MRQ Score"] * pcf, 2)
    else:
        next
    if "5 Year Sales Growth 5YA" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["5 Year Sales Growth 5YA Score"] * fiverev, 2)
    else:
        next
    if "Gross Margin 5YA" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Gross Margin 5YA Score"] * fivemar, 2)
    else:
        next
    if "P/E Ratio TTM" in factors:
        filtered_df['Combined Score'] = round(filtered_df['Combined Score'] + filtered_df["P/E Ratio TTM Score"] * pe,
                                              2)
    else:
        next
    if "Price to Book MRQ" in factors:
        filtered_df['Combined Score'] = round(
            filtered_df['Combined Score'] + filtered_df["Price to Book MRQ Score"] * pb, 2)
    else:
        next

    filtered_df = filtered_df.sort_values('Combined Score', ascending=False)
    filtered_df = filtered_df[list(np.append(cols, 'Combined Score'))]
    # subset = df[df['ticker'] == ticker].copy()
    filtered_df = filtered_df.head(9)

    if 'Market Cap' in filtered_df.columns:
        filtered_df = filtered_df.drop('Market Cap', axis=1)
    table = dbc.Table.from_dataframe(filtered_df)
    table1 = filtered_df['Ticker'].to_json(date_format='iso', orient='split')
    table2 = filtered_df.to_json()
    return table, table1, table2
    # filtered_df.to_json(date_format='iso', orient='split')
    # df_no_nan = df.dropna(subset = ['Company', 'Ticker']).copy()
    # df_no_nan.sort_values(['country'], inplace = True)


@app.callback(
    Output(component_id='price_plot', component_property='figure'),
    Input(component_id='my_output2', component_property='data'),
    Input(component_id='my_dates', component_property='start_date'),
    Input(component_id='my_dates', component_property='end_date'), prevent_initial_callback=True)
def create_graph(df_f, starts, ends):
    #years_ago = datetime.now() - relativedelta(years=1)

    df_l = pd.read_json(df_f)
    # , orient='split')

    # pd.read_json(df_f, orient='split')
    tickers = df_l['data']
    # tickers = df_l['Ticker']
    # tickers = df_f.values()
    # tickers = list(df_f.values)
    # df = pd.DataFrame(tickers)

    # start = years_ago
    # end = datetime.datetime.now()

    dfe = web.DataReader(tickers,
                         'yahoo', start=starts, end=ends)

    dfe = dfe.stack().reset_index().sort_values('Date')
    dfe['Daily Percentage Return'] = dfe['Close'].groupby(dfe['Symbols']).pct_change()

    # .(dffs['Close'] / dffs['Close'].shift(1))-1
    dfe['Cumulative Percentage Return'] = dfe['Daily Percentage Return'].groupby(dfe['Symbols']).apply(
        lambda x: x.add(1)).groupby(dfe['Symbols']).cumprod().fillna(0).groupby(dfe['Symbols']).apply(
        lambda x: x.sub(1))

    dfe = dfe.dropna()
    sumreturns = dfe['Cumulative Percentage Return'].groupby(dfe['Date']).mean()
    sumreturns = sumreturns.reset_index()

    # dffs = dfe[dfe['Symbols'].isin(tickers)]

    fig_close = px.line(dfe, x='Date', y='Cumulative Percentage Return', color='Symbols')
    fig_close.update_layout(
        yaxis_title='Returns',
        xaxis_title='Time',
        title='Cumulative Return Strategy',
        title_x=0.5,
        # title_y = 0.5,
        margin={'l': 0, 'r': 0},
        font=dict(
            family="Open Sans','sans-serif",
            size=15,
            color="black")
    )
    fig_close.update_xaxes(showgrid=False)
    fig_close.update_yaxes(tickformat=".1%")

    fig_close.add_trace(go.Scatter(x=sumreturns['Date'], y=sumreturns['Cumulative Percentage Return'], name="Strategy"))
    fig_close.layout.plot_bgcolor = 'white'
    return fig_close


@app.callback(
    Output(component_id='sector_pie', component_property='figure'),
    Input(component_id='my_output3', component_property='data'), prevent_initial_callback=True)
# This dataframe has 244 lines, but 4 distinct values for `day`
def create_sector_pie(dat):
    df_l = pd.read_json(dat).dropna()
    # df = px.data.tips()
    fig = px.pie(df_l['Sector'], values=df_l['Sector'].value_counts(), names=df_l['Sector'].unique())
    fig.update_layout(
        title='Distribution of Sectors',
        title_x=0.5,
        # title_y = 0.5,
        margin={'l': 0, 'r': 0},
        font=dict(
            family="Open Sans','sans-serif",
            size=15,
            color="black"))
    return fig

    
if __name__ == '__main__': 
    app.run_server(debug=True)