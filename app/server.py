from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://drive.google.com/uc?export=download&id=1-5oUidl5deUjNPtUGtxyNwIF-yi-3n7E'
export_file_name = '297 Butterrfly Resnet50 epo50 20mar19.pkl'

classes = ['Abisara bifasciata',
 'Acraea terpsicore',
 'Acytolepis lilacea',
 'Acytolepis puspa',
 'Aeromachus dubius',
 'Aeromachus pygmaeus',
 'Amblypodia anita',
 'Ampittia dioscorides',
 'Ancema blanka',
 'Anthene emolus',
 'Anthene lycaenina',
 'Appias albina',
 'Appias indra',
 'Appias lalage',
 'Appias libythea',
 'Appias lyncida',
 'Appias wardii',
 'Argynnis hybrida',
 'Arhopala abseus',
 'Arhopala alea',
 'Arhopala amantes',
 'Arhopala bazaloides',
 'Arhopala centaurus',
 'Ariadne ariadne',
 'Ariadne merione',
 'Arnetta mercara',
 'Arnetta vindhiana',
 'Athyma inara',
 'Athyma perius',
 'Athyma ranga',
 'Athyma selenophora',
 'Azanus jesous',
 'Azanus ubaldus',
 'Badamia exclamationis',
 'Baoris farri',
 'Baracus hampsoni',
 'Baracus subditus',
 'Belenois aurota',
 'Bibasis sena',
 'Bindahara moorei',
 'Borbo cinnara',
 'Burara gomata',
 'Burara jaina',
 'Byblia ilithyia',
 'Caleta decidia',
 'Caltoris canaraica',
 'Caltoris kumara',
 'Caltoris philippina',
 'Caprona agama',
 'Caprona alida',
 'Caprona ransonnettii',
 'Castalius rosimon',
 'Catapaecilma major',
 'Catochrysops strabo',
 'Catopsilia pomona',
 'Catopsilia pyranthe',
 'Celaenorrhinus ambareesa',
 'Celaenorrhinus fusca',
 'Celaenorrhinus leucocera',
 'Celaenorrhinus putra',
 'Celastrina lavendularis',
 'Celatoxia albidisca',
 'Cephrenes acalle',
 'Cepora nadina',
 'Cepora nerissa',
 'Cethosia mahratta',
 'Charaxes agrarius',
 'Charaxes bernardus',
 'Charaxes bharata',
 'Charaxes solon',
 'Cheritra freja',
 'Chilades lajus',
 'Chilades pandava',
 'Chilades parrhasius',
 'Choaspes benjaminii',
 'Cirrochroa thais',
 'Coladenia indrani',
 'Colias nilagiriensis',
 'Colotis amata',
 'Colotis aurora',
 'Colotis danae',
 'Colotis etrida',
 'Colotis fausta',
 'Creon cleobis',
 'Cupha erymanthis',
 'Cupitha purreea',
 'Curetis acuta',
 'Curetis siva',
 'Curetis thetis',
 'Cyrestis thyodamas',
 'Danaus chrysippus',
 'Danaus genutia',
 'Delias eucharis',
 'Deudorix epijarbas',
 'Discolampa ethion',
 'Discophora lepida',
 'Doleschallia bisaltide',
 'Dophla evelina',
 'Elymnias caudata',
 'Erionota torus',
 'Euchrysops cnejus',
 'Euploea core',
 'Euploea klugii',
 'Euploea sylvester',
 'Eurema andersonii',
 'Eurema blanda',
 'Eurema brigitta',
 'Eurema hecabe',
 'Eurema laeta',
 'Euripus consimilis',
 'Euthalia aconthea',
 'Euthalia lubentina',
 'Everes lacturnus',
 'Freyeria putli',
 'Freyeria trochylus',
 'Gangara thyrsis',
 'Gerosis bhagava',
 'Gomalia elma',
 'Graphium agamemnon',
 'Graphium antiphates',
 'Graphium doson',
 'Graphium nomius',
 'Graphium teredon',
 'Halpe hindu',
 'Halpe porus',
 'Halpemorpha hyrtacus',
 'Hasora badra',
 'Hasora chromus',
 'Hasora taminatus',
 'Hasora vitta',
 'Hebomoia glaucippe',
 'Horaga onyx',
 'Hyarotis adrastus',
 'Hypolimnas bolina',
 'Hypolimnas misippus',
 'Hypolycaena nilgirica',
 'Hypolycaena othona',
 'Iambrix salsala',
 'Idea malabarica',
 'Ionolyce helicon',
 'Iraota timoleon',
 'Ixias marianne',
 'Ixias pyrene',
 'Jamides alecto',
 'Jamides bochus',
 'Jamides celeno',
 'Junonia almana',
 'Junonia atlites',
 'Junonia hierta',
 'Junonia iphita',
 'Junonia lemonias',
 'Junonia orithya',
 'Kallima horsfieldii',
 'Kaniska canace',
 'Lampides boeticus',
 'Lasippa viraja',
 'Leptosia nina',
 'Leptotes plinius',
 'Lethe drypetis',
 'Lethe europa',
 'Lethe rohria',
 'Libythea laius',
 'Libythea myrrha',
 'Loxura atymnus',
 'Matapa aria',
 'Megisba malaya',
 'Melanitis leda',
 'Melanitis phedima',
 'Melanitis zitenius',
 'Moduza procris',
 'Mycalesis anaxias',
 'Mycalesis junonia',
 'Mycalesis mineus',
 'Mycalesis perseus',
 'Mycalesis subdita',
 'Mycalesis visala',
 'Nacaduba beroe',
 'Nacaduba hermus',
 'Nacaduba kurava',
 'Nacaduba pactolus',
 'Neopithecops zalmora',
 'Neptis clinia',
 'Neptis hylas',
 'Neptis jumbah',
 'Neptis nata',
 'Neptis palnica',
 'Notocrypta curvifascia',
 'Notocrypta paralysos',
 'Odontoptilum angulata',
 'Oriens concinna',
 'Oriens goloides',
 'Orsotriaena medus',
 'Pachliopta aristolochiae',
 'Pachliopta hector',
 'Pachliopta pandiyana',
 'Pantoporia hordonia',
 'Pantoporia sandaka',
 'Papilio buddha',
 'Papilio clytia',
 'Papilio crino',
 'Papilio demoleus',
 'Papilio dravidarum',
 'Papilio helenus',
 'Papilio liomedon',
 'Papilio paris',
 'Papilio polymnestor',
 'Papilio polytes',
 'Parantica aglea',
 'Parantica nilgiriensis',
 'Parantirrhoea marshalli',
 'Pareronia ceylanica',
 'Pareronia hippia',
 'Parnara spp',
 'Parthenos sylvia',
 'Pelopidas conjuncta',
 'Pelopidas mathias',
 'Pelopidas subochracea',
 'Petrelaea dana',
 'Phaedyma columella',
 'Phalanta alcippe',
 'Phalanta phalantha',
 'Pieris canidia',
 'Polytremis lubricans',
 'Pratapa deva',
 'Prioneris sita',
 'Prosotas dubiosa',
 'Prosotas nora',
 'Prosotas noreia',
 'Pseudocoladenia dan',
 'Pseudozizeeria maha',
 'Psolos fuligo',
 'Quedara basiflava',
 'Rachana jalindra',
 'Rapala iarbus',
 'Rapala lankana',
 'Rapala manea',
 'Rapala varuna',
 'Rathinda amor',
 'Rohana parisatis',
 'Salanoemia sala',
 'Sarangesa dasahara',
 'Sarangesa purendra',
 'Spalgis epius',
 'Spialia galba',
 'Spindasis ictis',
 'Spindasis lohita',
 'Spindasis schistacea',
 'Spindasis vulcanus',
 'Suastus gremius',
 'Surendra quercetorum',
 'Symphaedra nais',
 'Tagiades gana',
 'Tagiades japetus',
 'Tagiades litigiosa',
 'Tajuria cippus',
 'Tajuria jehana',
 'Talicada nyseus',
 'Tanaecia lepidea',
 'Tapena thwaitesi',
 'Taractrocera ceramas',
 'Taractrocera maevius',
 'Tarucus ananda',
 'Tarucus nara',
 'Telicota bambusae',
 'Telicota colon',
 'Telinga adolphei',
 'Telinga davisoni',
 'Telinga oculus',
 'Thaduka multicaudata',
 'Thoressa astigmata',
 'Thoressa evershedi',
 'Thoressa honorei',
 'Tirumala limniace',
 'Tirumala septentrionis',
 'Troides minos',
 'Udara akasa',
 'Udaspes folus',
 'Vanessa cardui',
 'Vanessa indica',
 'Vindula erota',
 'Virachola isocrates',
 'Virachola perse',
 'Ypthima asterope',
 'Ypthima baldus',
 'Ypthima ceylonica',
 'Ypthima chenu',
 'Ypthima huebneri',
 'Ypthima striata',
 'Ypthima ypthimoides',
 'Zeltus amasa',
 'Zesius chrysomallus',
 'Zinaspa todara',
 'Zipaetis saitis',
 'Zizeeria karsandra',
 'Zizina otis',
 'Zizula hylax',
 'Zographetus ogygia']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
