# Importing essential libraries and modules

from xml.etree.ElementTree import PI
from flask import Flask, render_template,request, Markup
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import joblib
import xgboost
import json
from difflib import SequenceMatcher
# from utils.disease import disease_dic
# from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
# import io
# import torch
# from torchvision import transforms
# from PIL import Image
# from utils.model import ResNet9
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt


rain_data = pd.read_excel("district wise rainfall normal.xlsx")
#print(rain_data["STATE_UT_NAME"])
# Loading crop recommendation model

crop_recommendation_model_path = 'ML models/XGBoost_recomendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))
yield_prediction = joblib.load('ML models\\random_forest_yield_predict_exclude_area.pkl')
cp = joblib.load("ML models\\random_forest_price_predict.pkl")
# recommend = np.array([[6,22,15,10.164313,91.223210,6.465913,106.362551]])
# yield_data = np.array([[2013,1,8,13.0,20.176277,20.620261,0]])
# price_data = np.array(([[1,5,5,3,40]]))
# print(crop_recommendation_model.predict(recommend))
# print(yield_prediction.predict(yield_data))
# print(cp.predict(price_data))
crop_list = ['Arhar/Tur','Bajra','Banana','Castor seed','Cotton(lint)','Gram','Grapes','Groundnut','Jowar','Linseed','Maize','Mango','Moong(Green Gram)','Niger seed','Onion','Rabi pulses','Cereals & Millets','Kharif pulses','Ragi','Rapeseed &Mustard','Rice','Safflower','Sesamum','Small millets','Soyabean','Sugarcane','Sunflower','Tobacco','Tomato','Urad','Wheat','oilseeds']

states_list=['Andaman and Nicobar Islands','Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chandigarh','Chhattisgarh','Dadra and Nagar Haveli','Goa','Gujarat','Haryana','Himachal Pradesh','Jammu and Kashmir ','Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra','Manipur','Meghalaya' 'Mizoram','Nagaland','Odisha','Puducherry','Punjab','Rajasthan','Sikkim','Tamil Nadu','Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal']

district_list=['24 PARAGANAS NORTH', '24 PARAGANAS SOUTH', 'ADILABAD','AGAR MALWA', 'AGRA', 'AHMADABAD', 'AHMEDNAGAR', 'AIZAWL', 'AJMER','AKOLA', 'ALAPPUZHA', 'ALIGARH', 'ALIRAJPUR', 'ALLAHABAD','ALMORA', 'ALWAR', 'AMBALA', 'AMBEDKAR NAGAR', 'AMETHI','AMRAVATI', 'AMRELI', 'AMRITSAR', 'AMROHA', 'ANAND', 'ANANTAPUR','ANANTNAG', 'ANJAW', 'ANUGUL', 'ANUPPUR', 'ARARIA', 'ARIYALUR','ARWAL', 'ASHOKNAGAR', 'AURAIYA', 'AURANGABAD', 'AZAMGARH','BADGAM', 'BAGALKOT', 'BAGESHWAR', 'BAGHPAT', 'BAHRAICH', 'BAKSA','BALAGHAT', 'BALANGIR', 'BALESHWAR', 'BALLIA', 'BALOD','BALODA BAZAR', 'BALRAMPUR', 'BANAS KANTHA', 'BANDA', 'BANDIPORA','BENGALURU', 'BANKA', 'BANKURA', 'BANSWARA', 'BARABANKI','BARAMULLA', 'BARAN', 'BARDHAMAN', 'BAREILLY', 'BARGARH', 'BARMER','BARNALA', 'BARPETA', 'BARWANI', 'BASTAR', 'BASTI', 'BATHINDA','BEED', 'BEGUSARAI', 'BELGAUM', 'BELLARY', 'BEMETARA','BENGALURU URBAN', 'BETUL', 'BHADRAK', 'BHAGALPUR', 'BHANDARA','BHARATPUR', 'BHARUCH', 'BHAVNAGAR', 'BHILWARA', 'BHIND','BHIWANI', 'BHOJPUR', 'BHOPAL', 'BIDAR', 'BIJAPUR', 'BIJNOR','BIKANER', 'BILASPUR', 'BIRBHUM', 'BISHNUPUR', 'BOKARO','BONGAIGAON', 'BOUDH', 'BUDAUN', 'BULANDSHAHR', 'BULDHANA','BUNDI', 'BURHANPUR', 'BUXAR', 'CACHAR', 'CHAMARAJANAGAR','CHAMBA', 'CHAMOLI', 'CHAMPAWAT', 'CHAMPHAI', 'CHANDAULI','CHANDEL', 'CHANDIGARH', 'CHANDRAPUR', 'CHANGLANG', 'CHATRA','CHHATARPUR', 'CHHINDWARA', 'CHIKBALLAPUR', 'CHIKMAGALUR','CHIRANG', 'CHITRADURGA', 'CHITRAKOOT', 'CHITTOOR', 'CHITTORGARH','CHURACHANDPUR', 'CHURU', 'COIMBATORE', 'COOCHBEHAR', 'CUDDALORE','CUTTACK', 'DADRA AND NAGAR HAVELI', 'DAKSHIN KANNAD', 'DAMOH','DANG', 'DANTEWADA', 'DARBHANGA', 'DARJEELING', 'DARRANG', 'DATIA','DAUSA', 'DAVANGERE', 'DEHRADUN', 'DEOGARH', 'DEOGHAR', 'DEORIA','DEWAS', 'DHALAI', 'DHAMTARI', 'DHANBAD', 'DHAR', 'DHARMAPURI','DHARWAD', 'DHEMAJI', 'DHENKANAL', 'DHOLPUR', 'DHUBRI', 'DHULE','DIBANG VALLEY', 'DIBRUGARH', 'DIMA HASAO', 'DIMAPUR','DINAJPUR DAKSHIN', 'DINAJPUR UTTAR', 'DINDIGUL', 'DINDORI','DODA', 'DOHAD', 'DUMKA', 'DUNGARPUR', 'DURG', 'EAST DISTRICT','EAST GARO HILLS', 'EAST GODAVARI', 'EAST JAINTIA HILLS','EAST KAMENG', 'EAST KHASI HILLS', 'EAST SIANG', 'EAST SINGHBUM','ERNAKULAM', 'ERODE', 'ETAH', 'ETAWAH', 'FAIZABAD', 'FARIDABAD','FARIDKOT', 'FARRUKHABAD', 'FATEHABAD', 'FATEHGARH SAHIB','FATEHPUR', 'FAZILKA', 'FIROZABAD', 'FIROZEPUR', 'GADAG','GADCHIROLI', 'GAJAPATI', 'GANDERBAL', 'GANDHINAGAR', 'GANGANAGAR','GANJAM', 'GARHWA', 'GARIYABAND', 'GAUTAM BUDDHA NAGAR', 'GAYA','GHAZIABAD', 'GHAZIPUR', 'GIRIDIH', 'GOALPARA', 'GODDA','GOLAGHAT', 'GOMATI', 'GONDA', 'GONDIA', 'GOPALGANJ', 'GORAKHPUR','GULBARGA', 'GUMLA', 'GUNA', 'GUNTUR', 'GURDASPUR', 'GURGAON','GWALIOR', 'HAILAKANDI', 'HAMIRPUR', 'HANUMANGARH', 'HAPUR','HARDA', 'HARDOI', 'HARIDWAR', 'HASSAN', 'HATHRAS', 'HAVERI','HAZARIBAGH', 'HINGOLI', 'HISAR', 'HOOGHLY', 'HOSHANGABAD','HOSHIARPUR', 'HOWRAH', 'IDUKKI', 'IMPHAL EAST', 'IMPHAL WEST','INDORE', 'JABALPUR', 'JAGATSINGHAPUR', 'JAIPUR', 'JAISALMER','JAJAPUR', 'JALANDHAR', 'JALAUN', 'JALGAON', 'JALNA', 'JALORE','JALPAIGURI', 'JAMMU', 'JAMNAGAR', 'JAMTARA', 'JAMUI','JANJGIR-CHAMPA', 'JASHPUR', 'JAUNPUR', 'JEHANABAD', 'JHABUA','JHAJJAR', 'JHALAWAR', 'JHANSI', 'JHARSUGUDA', 'JHUNJHUNU', 'JIND','JODHPUR', 'JORHAT', 'JUNAGADH', 'KABIRDHAM', 'KACHCHH', 'KADAPA','KAIMUR (BHABUA)', 'KAITHAL', 'KALAHANDI', 'KAMRUP','KAMRUP METRO', 'KANCHIPURAM', 'KANDHAMAL', 'KANGRA', 'KANKER','KANNAUJ', 'KANNIYAKUMARI', 'KANNUR', 'KANPUR DEHAT','KANPUR NAGAR', 'KAPURTHALA', 'KARAIKAL', 'KARAULI','KARBI ANGLONG', 'KARGIL', 'KARIMGANJ', 'KARIMNAGAR', 'KARNAL','KARUR', 'KASARAGOD', 'KASGANJ', 'KATHUA', 'KATIHAR', 'KATNI','KAUSHAMBI', 'KENDRAPARA', 'KENDUJHAR', 'KHAGARIA', 'KHAMMAM','KHANDWA', 'KHARGONE', 'KHEDA', 'KHERI', 'KHORDHA', 'KHOWAI','KHUNTI', 'KINNAUR', 'KIPHIRE', 'KISHANGANJ', 'KISHTWAR', 'KODAGU','KODERMA', 'KOHIMA', 'KOKRAJHAR', 'KOLAR', 'KOLASIB', 'KOLHAPUR','KOLLAM', 'KONDAGAON', 'KOPPAL', 'KORAPUT', 'KORBA', 'KOREA','KOTA', 'KOTTAYAM', 'KOZHIKODE', 'KRISHNA', 'KRISHNAGIRI','KULGAM', 'KULLU', 'KUPWARA', 'KURNOOL', 'KURUKSHETRA','KURUNG KUMEY', 'KUSHI NAGAR', 'LAHUL AND SPITI', 'LAKHIMPUR','LAKHISARAI', 'LALITPUR', 'LATEHAR', 'LATUR', 'LAWNGTLAI','LEH LADAKH', 'LOHARDAGA', 'LOHIT', 'LONGDING', 'LONGLENG','LOWER DIBANG VALLEY', 'LOWER SUBANSIRI', 'LUCKNOW', 'LUDHIANA','LUNGLEI', 'MADHEPURA', 'MADHUBANI', 'MADURAI', 'MAHARAJGANJ','MAHASAMUND', 'MAHBUBNAGAR', 'MAHE', 'MAHENDRAGARH', 'MAHESANA','MAHOBA', 'MAINPURI', 'MALAPPURAM', 'MALDAH', 'MALKANGIRI','MAMIT', 'MANDI', 'MANDLA', 'MANDSAUR', 'MANDYA', 'MANSA','MARIGAON', 'MATHURA', 'MAU', 'MAYURBHANJ', 'MEDAK','MEDINIPUR EAST', 'MEDINIPUR WEST', 'MEERUT', 'MEWAT', 'MIRZAPUR','MOGA', 'MOKOKCHUNG', 'MON', 'MORADABAD', 'MORENA', 'MUKTSAR','MUNGELI', 'MUNGER', 'MURSHIDABAD', 'MUZAFFARNAGAR', 'MUZAFFARPUR','MYSORE', 'NABARANGPUR', 'NADIA', 'NAGAON', 'NAGAPATTINAM','NAGAUR', 'NAGPUR', 'NAINITAL', 'NALANDA', 'NALBARI', 'NALGONDA','NAMAKKAL', 'NAMSAI', 'NANDED', 'NANDURBAR', 'NARAYANPUR','NARMADA', 'NARSINGHPUR', 'NASHIK', 'NAVSARI', 'NAWADA','NAWANSHAHR', 'NAYAGARH', 'NEEMUCH', 'NICOBARS', 'NIZAMABAD','NORTH AND MIDDLE ANDAMAN', 'NORTH DISTRICT', 'NORTH GARO HILLS','NORTH GOA', 'NORTH TRIPURA', 'NUAPADA', 'OSMANABAD', 'PAKUR','PALAKKAD', 'PALAMU', 'PALGHAR', 'PALI', 'PALWAL', 'PANCH MAHALS','PANCHKULA', 'PANIPAT', 'PANNA', 'PAPUM PARE', 'PARBHANI','PASHCHIM CHAMPARAN', 'PATAN', 'PATHANAMTHITTA', 'PATHANKOT','PATIALA', 'PATNA', 'PAURI GARHWAL', 'PERAMBALUR', 'PEREN', 'PHEK','PILIBHIT', 'PITHORAGARH', 'PONDICHERRY', 'POONCH', 'PORBANDAR','PRAKASAM', 'PRATAPGARH', 'PUDUKKOTTAI', 'PULWAMA', 'PUNE','PURBI CHAMPARAN', 'PURI', 'PURNIA', 'PURULIA', 'RAE BARELI','RAICHUR', 'RAIGAD', 'RAIGARH', 'RAIPUR', 'RAISEN', 'RAJAURI','RAJGARH', 'RAJKOT', 'RAJNANDGAON', 'RAJSAMAND', 'RAMANAGARA','RAMANATHAPURAM', 'RAMBAN', 'RAMGARH', 'RAMPUR', 'RANCHI','RANGAREDDI', 'RATLAM', 'RATNAGIRI', 'RAYAGADA', 'REASI', 'REWA','REWARI', 'RI BHOI', 'ROHTAK', 'ROHTAS', 'RUDRA PRAYAG','RUPNAGAR', 'S.A.S NAGAR', 'SABAR KANTHA', 'SAGAR', 'SAHARANPUR','SAHARSA', 'SAHEBGANJ', 'SAIHA', 'SALEM', 'SAMASTIPUR', 'SAMBA','SAMBALPUR', 'SAMBHAL', 'SANGLI', 'SANGRUR', 'SANT KABEER NAGAR','SANT RAVIDAS NAGAR', 'SARAIKELA KHARSAWAN', 'SARAN', 'SATARA','SATNA', 'SAWAI MADHOPUR', 'SEHORE', 'SENAPATI', 'SEONI','SEPAHIJALA', 'SERCHHIP', 'SHAHDOL', 'SHAHJAHANPUR', 'SHAJAPUR','SHAMLI', 'SHEIKHPURA', 'SHEOHAR', 'SHEOPUR', 'SHIMLA', 'SHIMOGA','SHIVPURI', 'SHOPIAN', 'SHRAVASTI', 'SIDDHARTH NAGAR', 'SIDHI','SIKAR', 'SIMDEGA', 'SINDHUDURG', 'SINGRAULI', 'SIRMAUR', 'SIROHI','SIRSA', 'SITAMARHI', 'SITAPUR', 'SIVAGANGA', 'SIVASAGAR', 'SIWAN','SOLAN', 'SOLAPUR', 'SONBHADRA', 'SONEPUR', 'SONIPAT', 'SONITPUR','SOUTH ANDAMANS', 'SOUTH DISTRICT', 'SOUTH GARO HILLS','SOUTH GOA', 'SOUTH TRIPURA', 'SOUTH WEST GARO HILLS','SOUTH WEST KHASI HILLS', 'SPSR NELLORE', 'SRIKAKULAM', 'SRINAGAR','SUKMA', 'SULTANPUR', 'SUNDARGARH', 'SUPAUL', 'SURAJPUR', 'SURAT','SURENDRANAGAR', 'SURGUJA', 'TAMENGLONG', 'TAPI', 'TARN TARAN','TAWANG', 'TEHRI GARHWAL', 'THANE', 'THANJAVUR', 'THE NILGIRIS','THENI', 'THIRUVALLUR', 'THIRUVANANTHAPURAM', 'THIRUVARUR','THOUBAL', 'THRISSUR', 'TIKAMGARH', 'TINSUKIA', 'TIRAP','TIRUCHIRAPPALLI', 'TIRUNELVELI', 'TIRUPPUR', 'TIRUVANNAMALAI','TONK', 'TUENSANG', 'TUMKUR', 'TUTICORIN', 'UDAIPUR', 'UDALGURI','UDAM SINGH NAGAR', 'UDHAMPUR', 'UDUPI', 'UJJAIN', 'UKHRUL','UMARIA', 'UNA', 'UNAKOTI', 'UNNAO', 'UPPER SIANG','UPPER SUBANSIRI', 'UTTARA KANNADA', 'UTTAR KASHI', 'VADODARA','VAISHALI', 'VALSAD', 'VARANASI', 'VELLORE', 'VIDISHA','VILLUPURAM', 'VIRUDHUNAGAR', 'VISAKHAPATANAM', 'VIZIANAGARAM','WARANGAL', 'WARDHA', 'WASHIM', 'WAYANAD', 'WEST DISTRICT','WEST GARO HILLS', 'WEST GODAVARI', 'WEST JAINTIA HILLS','WEST KAMENG', 'WEST KHASI HILLS', 'WEST SIANG', 'WEST SINGHBHUM','WEST TRIPURA', 'WOKHA', 'YADGIR', 'YAMUNANAGAR', 'YANAM','YAVATMAL', 'ZUNHEBOTO']

crops_price_list=['Arecanut','Arhar/Tur','Bajra','Banana','Barley','Bhindi','Black pepper','Blackgram', 'Bottle Gourd', 'Brinjal' ,'Cabbage', 'Cardamom' ,'Carrot','Cashewnut','Castor seed','Cauliflower','Coconut ','Coriander','Cotton(lint)','Cowpea(Lobia)','Cucumber','Dry chillies','Dry ginger','Garlic','Gram','Grapes','Groundnut','Horse-gram','Jowar','Jute','Lemon','Linseed','Maize','Mango','Masoor','Moong(Green Gram)','Niger seed','Onion','Orange','Papaya','Peas & beans(Pulses)','Potato','Ragi','Rapeseed &Mustard','Rice','Safflower','Sesamum','Soyabean','Sweet potato','Tea','Tobacco','Tomato','Turmeric','Urad','Wheat']
print(len(states_list))
print(len(district_list))
# Postal_code = input("enter postal code")
def weather_fetch(Postal_code):
    location_url = "https://api.worldpostallocations.com/pincode?postalcode="+str(Postal_code)+"&countrycode=IN"
    response1 = requests.get(location_url)
    response1_data= response1.json()
    location_data=response1_data["result"][0]
    postalLocation = location_data['postalLocation']
    province = location_data["province"]
    latitude = location_data['latitude']
    longitude = location_data['longitude']
    state = location_data["state"]
    district = location_data["district"]
    d=district.replace("a ","+")

    #print(postalLocation,province,latitude,longitude)

    weather_api = "http://api.openweathermap.org/data/2.5/forecast?lat="+str(latitude)+"&lon="+str(longitude)+"&appid=" #enter your API id here
    response2 = requests.get(weather_api)
    response2_data = response2.json()
    weather_data = response2_data["list"][0]
    temperature = (float(weather_data["main"]["temp"])-273.15)
    pressure = weather_data["main"]['pressure']
    humidity = weather_data["main"]["humidity"]
    wind_speed = weather_data["wind"]["speed"]
    # print(response2_data)

    # return(temperature,pressure,humidity,wind_speed,province)
    # print(state)
    # print(d)
    # rain_api = "https://datasource.kapsarc.org/api/records/1.0/search/?dataset=district-wise-rainfall-data-for-india-2014&q=&facet=date&facet=districts_name&facet=indian_states_name&facet=variable_name&refine.date=2020&refine.indian_states_name="+state+"&refine.districts_name="+d
    # response3 = requests.get(rain_api)
    # response3_data = response3.json()
    # print(response3_data)
    df1 = rain_data[rain_data["STATE_UT_NAME"]==state.upper()]
    for index, row in df1.iterrows():
        a = SequenceMatcher(None, district.upper(),row["DISTRICT"]) 
        print(district)
        print(row["DISTRICT"])
        print(a.ratio())
        print(row["Avarage"])
        if a.ratio() > .8:
            rain_fall=row["Avarage"]
            break
            

        else :
            rain_fall = 123
            print("rain fall information is not available for ",district)
    print(rain_fall)       
    return(temperature,pressure,humidity,wind_speed,province,rain_fall,state,district)
#weather_fetch(Postal_code)
    
    

    # """
    # Fetch and returns the temperature and humidity of a city
    # :params: city_name
    # :return: temperature, humidity
    # """
    # api_key = config.weather_api_key
    # base_url = "http://api.openweathermap.org/data/2.5/weather?"

    # complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    # response = requests.get(complete_url)
    # x = response.json()

    # if x["cod"] != "404":
    #     y = x["main"]

    #     temperature = round((y["temp"] - 273.15), 2)
    #     humidity = y["humidity"]
    #     return temperature, humidity
    # else:
    #     return None


# # ===============================================================================================
# # ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    #return "hello"
    title = 'Crop harvest'
    return render_template('index.html',title='Crop harvest')

#render crop recommendation form page


@ app.route('/crop_recommend',methods=['GET', 'POST'])
def crop_recommend():
    import requests
    import pandas as pd
    # data=requests.get("https://api.thingspeak.com/channels/1733360/feeds.json?api_key=EM4OVKFQ02IBZE2K&results=2")
    # n=data.json()['feeds'][-1]['field3']
    # p=data.json()['feeds'][-1]['field4']
    # k=data.json()['feeds'][-1]['field5']
    # temp=data.json()['feeds'][-1]['field2']
    title = 'Crop Recommendation'
    return render_template('crop_recommend.html', title=title)   #, n=n, p=p, k=k, temp=temp)

# render fertilizer recommendation form page


@ app.route('/yeild', methods=['GET', 'POST'])
def yeild():
    import requests
    import pandas as pd
    # data=requests.get("https://api.thingspeak.com/channels/1733360/feeds.json?api_key=EM4OVKFQ02IBZE2K&results=2")
    # temp=data.json()['feeds'][-1]['field2']
    # hum=data.json()['feeds'][-1]['field1']
    title = 'crop yeild prediction'

    return render_template('crop_yeild.html', title=title  )   ##, temp=temp, hum=hum)

# # render price input page
@app.route('/crop_price', methods=['GET', 'POST'])
def crop_price():
    #return "this is crop prediction page"
    title = 'crop price'
    return render_template('crop_price.html', title=title)





# # ===============================================================================================

# # RENDER PREDICTION PAGES

# # render crop recommendation result page


@ app.route('/crop_predict', methods=['POST'])
def crop_predict():
    title = 'Crop Recommended'

    if request.method == 'POST':
        PIN = request.form["Pin_code"]
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])
        ph = float(request.form['ph'])
        #rainfall = float(request.form['rainfall'])

        temperature,pressure,humidity,wind_speed,province,rainfall,state,district = weather_fetch(PIN)
        value1 = np.array([[N,P,K,temperature,float(humidity),ph,float(rainfall)]])
        crop_predict = crop_recommendation_model.predict(value1)[0]



        # state=request.form['stt']
        # city = request.form['city']

        
    #     if weather_fetch(city) != None:
    #         temperature, humidity = weather_fetch(city)
    #         data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    #         my_prediction = crop_recommendation_model.predict(data)
    #         final_prediction = my_prediction[0]

    return render_template('crop-result.html', prediction=crop_predict ,city=province,temper=round(temperature,2),rain=round(rainfall*10,2),title=title)
        # else:
        #     return render_template('try_again.html', title=title)
# # render fertilizer recommendation result page
@ app.route('/yeild-predict',methods=['POST'])
def yeild_predict():
    title = 'yeild predicted'

    if request.method == 'POST':
        PIN = request.form['Pin_code']
        # district = request.form['city']
        year = int(request.form['year'])
        season = int(request.form['season'])
        crop = int(request.form['crop'])
        # Temperature = request.form['Temperature']
        # humidity= request.form['humidity']
        # soilmoisture= request.form['soilmoisture']
        area_s = float(request.form['area'])
        area_p = area_s*100
        soil_type=int(request.form["soil_type"])


        temperature,pressure,humidity,wind_speed,province,rainfall,state,district= weather_fetch(PIN)
        value2 = np.array([[year,season,crop,temperature,float(wind_speed),float(pressure),float(humidity),soil_type]])
        yieldh = yield_prediction.predict(value2)[0]
        yieldh = round(yieldh/100, 3)
        yield_area = round(yieldh*area_s, 4)
        
##        N = int(request.form['nitrogen'])
##        P = int(request.form['phosphorous'])
##        K = int(request.form['pottasium'])
##        ph = float(request.form['ph'])
##        rainfall = float(request.form['rainfall'])
##
##        # state = request.form.get("stt")
##        city = request.form.get("city")
##
##        if weather_fetch(city) != None:
##            temperature, humidity = weather_fetch(city)
##            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
##            my_prediction = crop_recommendation_model.predict(data)
##            final_prediction = my_prediction[0]

        return render_template('yeild_prediction.html',place = province,temper=round(temperature,2),rain=round(rainfall*10,2),area=round(area_s,1),hum=humidity, prediction=yield_area,crops = crop_list[crop],title=title)

#     return render_template('try_again.html', title=title)



# # render disease prediction result page


# @app.route('/crop_price', methods=['GET', 'POST'])
# def crop_price():
#     #return "this is crop prediction page"
#     title = 'crop price'
#     return render_template('crop_price.html', title=title)

@ app.route('/price_predict',methods=['POST'])
def price_predict():
    title = 'price Suggestion'
    if request.method == 'POST':
        PIN = request.form['Pin_code']
        year = int(request.form['year'])
        season = int(request.form['season'])
        crop = int(request.form['crop'])

        temperature,pressure,humidity,wind_speed,province,rainfall,state,district= weather_fetch(PIN)
        for index1,value1 in enumerate(states_list):
            b = SequenceMatcher(None, state.upper(),value1.upper())
            if b.ratio() > .9:
                state_num = index1
                print(value1)

        for index2,value2 in enumerate(district_list):
            c  = SequenceMatcher(None, district.upper(),value2.upper())
            if c.ratio() > .9:
                district_num = index2
                print(value2)


        p_result = cp.predict([[int(state_num),
                float(district_num),
                float(year),
                float(season),
                float(crop)]])

        return render_template('price_prediction.html', title=title, p_result=round(p_result[0],0),state_s=state,district_d = district,temp = round(temperature,2), crop_c = crops_price_list[crop])
    return render_template('try_again.html', title=title)



# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
