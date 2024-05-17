from dash import Dash, html ,dcc ,callback,Input,Output
import pickle
import numpy as np
import pandas as pd
import random
import sklearn
import tensorflow as tf
from tensorflow import keras
import plotly.express as px

app = Dash(__name__)
server = app.server

month_dict = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
dict_month = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

enc = pickle.load(open("/encoder.sav","rb"))
model = keras.models.load_model("/modle.keras")


app.layout = html.Div([
    html.H1(children='Climate Analysis Dashboard',style={'textAlign': 'center'}),
    html.Hr(),
    html.Div(children=[
        html.Div(children=[
        html.Div(children="Enter the City"),
        dcc.Dropdown(['Aberdeen', 'Abilene', 'Abohar', 'Achalpur', 'Adilabad', 'Adoni', 'Agartala', 'Agra', 'Ahmadabad', 'Ahmadnagar', 'Aizawl', 'Ajmer', 'Akola', 'Akron', 'Alandur', 'Alappuzha', 'Albuquerque', 'Alexandria', 'Aligarh', 'Allahabad', 'Allentown', 'Alwar', 'Amarillo', 'Ambala', 'Ambarnath', 'Ambattur', 'Ambur', 'Amravati', 'Amritsar', 'Amroha', 'Anaheim', 'Anand', 'Anantapur', 'Anchorage', 'Ann Arbor', 'Antioch', 'Ara', 'Arlington', 'Arvada', 'Asansol', 'Atlanta', 'Aurangabad', 'Aurora', 'Austin', 'Avadi', 'Azamgarh', 'Badlapur', 'Bagaha', 'Bahadurgarh', 'Baharampur', 'Bahraich', 'Baidyabati', 'Bakersfield', 'Baleshwar', 'Ballia', 'Bally', 'Baltimore', 'Balurghat', 'Banda', 'Bangalore', 'Bangaon', 'Bankura', 'Bansbaria', 'Barakpur', 'Baranagar', 'Barasat', 'Barddhaman', 'Bareli', 'Baripada', 'Barnala', 'Barsi', 'Basildon', 'Basirhat', 'Basti', 'Batala', 'Baton Rouge', 'Beaumont', 'Beawar', 'Begusarai', 'Belfast', 'Belgaum', 'Bellary', 'Bellevue', 'Berkeley', 'Bettiah', 'Bhadravati', 'Bhadreswar', 'Bhagalpur', 'Bharatpur', 'Bharuch', 'Bhatpara', 'Bhavnagar', 'Bhilai', 'Bhilwara', 'Bhimavaram', 'Bhind', 'Bhiwandi', 'Bhiwani', 'Bhopal', 'Bhubaneswar', 'Bhuj', 'Bhusawal', 'Bid', 'Bidar', 'Bihar', 'Bijapur', 'Bikaner', 'Bilaspur', 'Birmingham', 'Blackburn', 'Blackpool', 'Bokaro', 'Bolton', 'Bombay', 'Boston', 'Botad', 'Bournemouth', 'Bradford', 'Brahmapur', 'Bridgeport', 'Brighton', 'Bristol', 'Brownsville', 'Budaun', 'Buffalo', 'Bulandshahr', 'Burbank', 'Burhanpur', 'Calcutta', 'Cambridge', 'Cape Coral', 'Cardiff', 'Carrollton', 'Cary', 'Cedar Rapids', 'Champdani', 'Chandannagar', 'Chandausi', 'Chandigarh', 'Chandler', 'Chandrapur', 'Charleston', 'Charlotte', 'Chas', 'Chattanooga', 'Chelmsford', 'Cheltenham', 'Chesapeake', 'Chhapra', 'Chhatarpur', 'Chhindwara', 'Chicago', 'Chikmagalur', 'Chitradurga', 'Chittaurgarh', 'Chula Vista', 'Churu', 'Cincinnati', 'Clarksville', 'Clearwater', 'Cleveland', 'Colchester', 'Colorado Springs', 'Columbia', 'Columbus', 'Concord', 'Coral Springs', 'Corona', 'Corpus Christi', 'Costa Mesa', 'Coventry', 'Crawley', 'Cuddapah', 'Dallas', 'Damoh', 'Darbhanga', 'Darjiling', 'Dayton', 'Dehra Dun', 'Dehri', 'Delhi', 'Denton', 'Denver', 'Deoria', 'Derby', 'Des Moines', 'Detroit', 'Dewas', 'Dhanbad', 'Dharmavaram', 'Dhaulpur', 'Dhule', 'Dibrugarh', 'Dimapur', 'Dindigul', 'Downey', 'Dudley', 'Dum Dum', 'Dundee', 'Durg', 'Durgapur', 'Durham', 'East Los Angeles', 'Eastbourne', 'Edinburgh', 'Edison', 'El Monte', 'El Paso', 'Elizabeth', 'Eluru', 'Erode', 'Escondido', 'Etah', 'Etawah', 'Eugene', 'Evansville', 'Exeter', 'Fairfield', 'Faizabad', 'Faridabad', 'Farrukhabad', 'Fatehpur', 'Fayetteville', 'Firozabad', 'Firozpur', 'Flint', 'Fontana', 'Fort Collins', 'Fort Lauderdale', 'Fort Wayne', 'Fort Worth', 'Fremont', 'Fresno', 'Fullerton', 'Gadag', 'Gainesville', 'Gandhidham', 'Gandhinagar', 'Ganganagar', 'Gangapur', 'Gangawati', 'Garden Grove', 'Garland', 'Gaya', 'Ghaziabad', 'Ghazipur', 'Gilbert', 'Gillingham', 'Glasgow', 'Glendale', 'Gloucester', 'Godhra', 'Gonda', 'Gondal', 'Gorakhpur', 'Grand Prairie', 'Grand Rapids', 'Green Bay', 'Greensboro', 'Gudalur', 'Gudivada', 'Gulbarga', 'Guna', 'Guntakal', 'Guntur', 'Gurgaon', 'Guwahati', 'Gwalior', 'Habra', 'Hajipur', 'Haldia', 'Haldwani', 'Halisahar', 'Hampton', 'Hanumangarh', 'Haora', 'Hapur', 'Hardoi', 'Haridwar', 'Hartford', 'Hassan', 'Hathras', 'Hayward', 'Hazaribag', 'Henderson', 'Hialeah', 'Highlands Ranch', 'Hindupur', 'Hisar', 'Hollywood', 'Hoshangabad', 'Hoshiarpur', 'Hospet', 'Hosur', 'Houston', 'Hubli', 'Huddersfield', 'Huntington Beach', 'Huntsville', 'Hyderabad', 'Ichalkaranji', 'Imphal', 'Independence', 'Indianapolis', 'Indore', 'Inglewood', 'Ingraj Bazar', 'Ipswich', 'Irvine', 'Irving', 'Itarsi', 'Jabalpur', 'Jackson', 'Jacksonville', 'Jagadhri', 'Jaipur', 'Jalandhar', 'Jalna', 'Jalpaiguri', 'Jamalpur', 'Jammu', 'Jamnagar', 'Jamshedpur', 'Jamuria', 'Jaunpur', 'Jersey City', 'Jetpur', 'Jhansi', 'Jhunjhunun', 'Jind', 'Jodhpur', 'Joliet', 'Junagadh', 'Kaithal', 'Kakinada', 'Kalol', 'Kalyan', 'Kamarhati', 'Kanchipuram', 'Kanchrapara', 'Kanpur', 'Kansas City', 'Kapra', 'Karimnagar', 'Karnal', 'Kashipur', 'Katihar', 'Khammam', 'Khandwa', 'Khanna', 'Kharagpur', 'Khardaha', 'Khurja', 'Killeen', 'Kingston Upon Hull', 'Knoxville', 'Kochi', 'Kolar', 'Kolhapur', 'Kollam', 'Korba', 'Kota', 'Krishnanagar', 'Kulti', 'Kumbakonam', 'Lafayette', 'Lakewood', 'Lakhimpur', 'Lakhnau', 'Lalitpur', 'Lancaster', 'Lansing', 'Laredo', 'Las Vegas', 'Latur', 'Leeds', 'Leicester', 'Lexington Fayette', 'Lincoln', 'Little Rock', 'Liverpool', 'London', 'Long Beach', 'Loni', 'Los Angeles', 'Louisville', 'Lowell', 'Lubbock', 'Ludhiana', 'Luton', 'Machilipatnam', 'Madanapalle', 'Madhyamgram', 'Madison', 'Madras', 'Madurai', 'Mahbubnagar', 'Mahesana', 'Maisuru', 'Malegaon', 'Maler Kotla', 'Manchester', 'Mandsaur', 'Mandya', 'Mangaluru', 'Mathura', 'Mau', 'Memphis', 'Mesa', 'Mesquite', 'Metairie', 'Miami', 'Middlesbrough', 'Milwaukee', 'Minneapolis', 'Miramar', 'Mirzapur', 'Mobile', 'Modesto', 'Moga', 'Montgomery', 'Moradabad', 'Morena', 'Moreno Valley', 'Mormugao', 'Morvi', 'Motihari', 'Munger', 'Murwara', 'Muzaffarnagar', 'Muzaffarpur', 'Nadiad', 'Nagda', 'Nagercoil', 'Nagpur', 'Naihati', 'Nalgonda', 'Nanded', 'Nandurbar', 'Nandyal', 'Nangloi Jat', 'Naperville', 'Nashville', 'Navadwip', 'Navsari', 'New Delhi', 'New Haven', 'New Orleans', 'New York', 'Newark', 'Newcastle Upon Tyne', 'Newport', 'Newport News', 'Neyveli', 'Nizamabad', 'Nogales', 'Norfolk', 'Norman', 'North Las Vegas', 'Northampton', 'Norwalk', 'Norwich', 'Nottingham', 'Nuevo Laredo', 'Oakland', 'Oceanside', 'Oklahoma City', 'Olathe', 'Oldham', 'Omaha', 'Ongole', 'Ontario', 'Orai', 'Orange', 'Orlando', 'Overland Park', 'Oxford', 'Oxnard', 'Palakkad', 'Palanpur', 'Pali', 'Pallavaram', 'Palmdale', 'Palwal', 'Panihati', 'Panipat', 'Panvel', 'Paradise', 'Parbhani', 'Pasadena', 'Patan', 'Paterson', 'Pathankot', 'Patiala', 'Patna', 'Pembroke Pines', 'Peoria', 'Peterborough', 'Phagwara', 'Philadelphia', 'Phoenix', 'Pilibhit', 'Pimpri', 'Pittsburgh', 'Plano', 'Plymouth', 'Pomona', 'Pondicherry', 'Ponnani', 'Poole', 'Porbandar', 'Port Blair', 'Port Saint Lucie', 'Portland', 'Portsmouth', 'Preston', 'Proddatur', 'Providence', 'Provo', 'Pudukkottai', 'Pueblo', 'Pune', 'Puri', 'Purnia', 'Puruliya', 'Rae Bareli', 'Raichur', 'Raiganj', 'Raigarh', 'Raipur', 'Rajamahendri', 'Rajapalaiyam', 'Rajkot', 'Rajpur', 'Raleigh', 'Rampur', 'Ranchi', 'Rancho Cucamonga', 'Raniganj', 'Ratlam', 'Raurkela', 'Reading', 'Reno', 'Rewa', 'Rewari', 'Rialto', 'Richardson', 'Richmond', 'Rishra', 'Riverside', 'Robertsonpet', 'Rochester', 'Rockford', 'Rohtak', 'Roseville', 'Rotherham', 'Sacramento', 'Sagar', 'Saharanpur', 'Saharsa', 'Saint Helens', 'Saint Louis', 'Saint Paul', 'Saint Petersburg', 'Salem', 'Salinas', 'Salt Lake City', 'Sambalpur', 'Sambhal', 'San Antonio', 'San Bernardino', 'San Diego', 'San Francisco', 'San Jose', 'Santa Ana', 'Santa Clara', 'Santa Clarita', 'Santa Rosa', 'Sasaram', 'Satara', 'Satna', 'Savannah', 'Sawai Madhopur', 'Scottsdale', 'Seattle', 'Selam', 'Seoni', 'Shahjahanpur', 'Shantipur', 'Sheffield', 'Shiliguri', 'Shillong', 'Shimla', 'Shimoga', 'Sholapur', 'Shreveport', 'Shrirampur', 'Sikar', 'Silchar', 'Simi Valley', 'Sioux Falls', 'Sirsa', 'Sitapur', 'Siwan', 'Slough', 'Sonipat', 'South Bend', 'Southampton', 'Southend On Sea', 'Spokane', 'Spring Valley', 'Springfield', 'Srikakulam', 'Srinagar', 'Stamford', 'Sterling Heights', 'Stockport', 'Stockton', 'Stoke On Trent', 'Sultanpur', 'Sunderland', 'Sunnyvale', 'Sunrise Manor', 'Surat', 'Surendranagar', 'Suriapet', 'Sutton Coldfield', 'Swansea', 'Swindon', 'Syracuse', 'Tacoma', 'Tadepallegudem', 'Tallahassee', 'Tambaram', 'Tampa', 'Tempe', 'Tenali', 'Thana', 'Thanesar', 'Thanjavur', 'Thiruvananthapuram', 'Thornton', 'Thousand Oaks', 'Thrissur', 'Tiruchchirappalli', 'Tirunelveli', 'Tirupati', 'Tiruppur', 'Tiruvannamalai', 'Tiruvottiyur', 'Titagarh', 'Toledo', 'Toms River', 'Tonk', 'Topeka', 'Torrance', 'Tucson', 'Tulsa', 'Tumkur', 'Udgir', 'Udupi', 'Ujjain', 'Ulhasnagar', 'Ulubaria', 'Unnao', 'Vadodara', 'Vallejo', 'Vancouver', 'Varanasi', 'Vejalpur', 'Velluru', 'Veraval', 'Vidisha', 'Vijayawada', 'Virar', 'Virginia Beach', 'Visakhapatnam', 'Visalia', 'Vizianagaram', 'Waco', 'Walsall', 'Warangal', 'Wardha', 'Warren', 'Washington', 'Waterbury', 'Watford', 'West Bromwich', 'West Covina', 'West Jordan', 'West Valley City', 'Westminster', 'Wichita', 'Wichita Falls', 'Windsor', 'Winston Salem', 'Wolverhampton', 'Worcester', 'Yamunanagar', 'Yavatmal', 'Yelahanka', 'Yonkers', 'York'], 'Madras', multi=False , id='city')
    ],style={'padding-left':30,'padding-top':10}),
    html.Div([
        html.Div(children="Enter Year"),
        dcc.Input(placeholder='Enter a year...', type='number', value='1990' , id='year'),
    ],style={'padding-left':30,'padding-top':10})
    ]),
    html.Div([
        html.Div(children="Enter month"),
        dcc.Dropdown(['January','February','March','April','May','June','July','August','September','October','November','December'], 'January', multi=False , id='mon')
    ],style={'padding-left':30,'padding-top':10}),
    html.Div(children=[
        dcc.Input(type='text',value="",id='out1')
    ],style={'padding-left':30,'padding-top':10}),
    html.H2(children="Forecast Analysis",style={'textAlign': 'center'}),
    dcc.Graph(id="out2")
])

def preprocess(city,year,mon):
    y = int(year)
    cit = int(enc.transform([city])[0])
    mo = int(month_dict[mon])
    return y,cit,mo

def shaper(year,city,month):
    new_data = np.array([city,int(year),month])  
    new_data = new_data.reshape(1, 3)
    return new_data


def predict_with_model(data,month,city):
    p = model.predict(data)
    out = int(p[0][0])+(random.random()*10)+(random.choice([1,-1])*(random.random()*5))
    
    if(month=='March' or month=='April'or month=='May'):
        out += (random.random()*10)
    if(city=='London' or city=='Basildon'or month=='Shimla'):
        out -= (random.random()*10)
    return round(out)

def generate_dataframes(city,year,mon):
    yea,cit,mont = preprocess(city,year,mon)
    obj = shaper(yea,cit,mont)
    temp = predict_with_model(obj,mon,city)
    datum = [{"City": city, "Month": mon, "Year": year, "Temperature": temp}]
    df = pd.DataFrame(datum,index=[0])
    for i in range(mont,mont+11):
        nex = (i%12)+1
        nmon = dict_month[nex]
        nobj = shaper(yea,cit,nex)
        npred = predict_with_model(nobj,nmon,city)
        n1 = pd.DataFrame([{"City":city,"Month":nmon,"Year":year,"Temperature":npred}])
        df = pd.concat([df,n1],ignore_index=True)
    return df

def plot_graph(data):
    fig = px.line(data,"Month","Temperature",markers=True)
    return fig


@callback(
    # Output('out','value'),
    Output('out1','value'),
    Output('out2','figure'),
    Input('city','value'),
    Input('year','value'),
    Input('mon','value')
)
def update_all(city,year,mon):

    y = str(year)   #print with string
    cit = int(enc.transform([city])[0])
    val = str(cit)  #printing with string
    mo = int(month_dict[mon])
    v = str(mo)     #print with string


    new_data = np.array([cit,int(year),mo])  # This creates a 1D array
    new_data = new_data.reshape(1, 3)
    pred = predict_with_model(new_data,mon,city)
    data = generate_dataframes(city,year,mon)

    fig = plot_graph(data)
    

    ou = str(pred)

    # out = city+" "+y+" "+mon+" "
    return ou,fig

if __name__ == '__main__':
    app.run(debug=True)
