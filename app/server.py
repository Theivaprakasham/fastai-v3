from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://drive.google.com/uc?export=download&id=1-As7hDred2Vm08I0a_TgY1_SjQYtYE1T'
export_file_name = '298Resnet50new.pkl'

def top_3_accuracy(input:Tensor, targs:Tensor, k:int=3)->Rank0Tensor:
  n = targs.shape[0]
  input = input.topk(k=k, dim=-1)[1].view(n, -1)
  targs = targs.view(n,-1)
  return (input == targs).sum(dim=1, dtype=torch.float32).mean()

def top_5_accuracy(input:Tensor, targs:Tensor, k:int=5)->Rank0Tensor:
  n = targs.shape[0]
  input = input.topk(k=k, dim=-1)[1].view(n, -1)
  targs = targs.view(n,-1)
  return (input == targs).sum(dim=1, dtype=torch.float32).mean()

classes = ["Double-banded Judy (Abisara bifasciata)","Tawny Coster (Acraea terpsicore)","Hampson's Hedge Blue (Acytolepis lilacea)","Common Hedge Blue (Acytolepis puspa)","Dingy Scrub Hopper (Aeromachus dubius)","Pygmy Scrub Hopper (Aeromachus pygmaeus)","Leaf Blue (Amblypodia anita)","Bush Hopper (Ampittia dioscorides)","Silver Royal (Ancema blanka)","Common Ciliate Blue (Anthene emolus)","Pointed Ciliate Blue (Anthene lycaenina)","Common Albatross (Appias albina)","Plain Puffin (Appias indra)","Spot Puffin (Appias lalage)","Striped Albatross (Appias libythea)","Chocolate Albatross (Appias lyncida)","Lesser Albatross2 (Appias wardii)","Indian Fritillary (Argynnis hyperbius/Argynnis hybrida)","Aberrant Bushblue (Arhopala abseus)","Kanara Oakblue (Arhopala alea)","Large Oakblue (Arhopala amantes)","Tamil Oakblue (Arhopala bazaloides)","Centaur Oakblue (Arhopala centaurus)","Angled Castor (Ariadne ariadne)","Common Castor (Ariadne merione)","Coorg Forest Hopper (Arnetta mercara)","Vindhyan Bob (Arnetta vindhiana)","Colour Sergeant (Athyma inara)","Common Sergeant (Athyma perius)","Black-vein Sergeant (Athyma ranga)","Staff Sergeant (Athyma selenophora)","African Babul Blue (Azanus jesous)","Bright Babul Blue (Azanus ubaldus)","Brown Awl (Badamia exclamationis)","Paintbrush Swift (Baoris farri)","Hampson's Hedge Hopper (Baracus hampsoni)","Hedge Hopper (Baracus subditus)","Pioneer (Belenois aurota)","Orange-tail Awl (Bibasis sena)","Blue-edged Plane (Bindahara moorei)","Rice Swift (Borbo cinnara)","Pale Green Awlet (Burara gomata)","Orange Striped Awlet (Burara jaina)","Joker (Byblia ilithyia)","Angled Pierrot (Caleta decidia)","Kanara Swift (Caltoris canaraica)","Blank Swift (Caltoris kumara)","Philippine Swift (Caltoris philippina)","Spotted Angle (Caprona agama)","Alida Angle (Caprona alida)","Golden Angle (Caprona ransonnetti)","Common Pierrot (Castalius rosimon)","Common Tinsel (Catapaecilma major)","Forget-me-not (Catochrysops strabo)","Common Emigrant (Catopsilia pomona)","Mottled Emigrant (Catopsilia pyranthe)","Malabar Spotted Flat (Celaenorrhinus ambareesa)","Common Spotted Flat (Celaenorrhinus leucocera)","Bengal Spotted Flat (Celaenorrhinus putra)","Tamil Spotted Flat (Celaenorrhinus ruficornis/Celaenorrhinus fusca)","Plain Hedge Blue (Celastrina lavendularis)","White-disc Hedge Blue (Celatoxia albidisca)","Plain Palm-dart (Cephrenes acalle)","Lesser Gull (Cepora nadina)","Common Gull (Cepora nerissa)","Tamil Lacewing (Cethosia mahratta)","Indian Tawny Rajah (Charaxes psaphon)","Black Rajah (Charaxes solon)","Common Imperial (Cheritra freja)","Lime Blue (Chilades lajus)","Plains Cupid (Chilades pandava)","Small Cupid (Chilades parrhassius)","Orchid Tit (Chliaria othona/Hypolycaena othona)","Indian Awlking (Choaspes benjaminii)","Tamil Yeoman (Cirrochroa thais)","Tricoloured Pied Flat (Coladenia indrani)","Nilgiri Clouded Yellow (Colias nilagiriensis)","Small Salmon Arab (Colotis amata)","Plain Orange-tip (Colotis aurora)","Crimson-tip (Colotis danae)","Small Orange-tip (Colotis etrida)","Large Salmon Arab (Colotis fausta)","Broadtail Royal (Creon cleobis)","Rustic (Cupha erymanthis)","Wax Dart (Cupitha purreea)","Angled Sunbeam (Curetis acuta)","Siva Sunbeam (Curetis siva)","Indian Sunbeam (Curetis thetis)","Grey Count (Cynitia lepidea/Tanaecia lepidea)","Common Map (Cyrestis thyodamas)","Plain Tiger (Danaus chrysippus)","Striped Tiger (Danaus genutia)","Common Jezebel (Delias eucharis)","Cornelian (Deudorix epijarbas)","Banded Blue Pierrot (Discolampa ethion)","Southern Duffer (Discophora lepida)","Autumn Leaf (Doleschallia bisaltide)","Redspot Duke (Dophla evelina)","Southern Palmfly (Elymnias caudata)","Rounded Palm-redeye (Erionota torus)","Gram Blue (Euchrysops cnejus)","Common Crow (Euploea core)","Brown King Crow (Euploea klugii)","Double-branded Crow (Euploea sylvester)","One-spot Grass Yellow (Eurema andersoni)","Three-spot Grass Yellow (Eurema blanda)","Small Grass Yellow (Eurema brigitta)","Common Grass Yellow (Eurema hecabe)","Spotless Grass Yellow (Eurema laeta)","Painted Courtesan (Euripus consimilis)","Common Baron (Euthalia aconthea)","Gaudy Baron (Euthalia lubentina)","Indian Cupid (Everus lacturnus)","Small Grass Jewel (Freyeria putli)","Grass Jewel (Freyeria trochylus)","Giant Redeye (Gangara thyrsis)","Common Yellow-breasted Flat (Gerosis bhagava)","African Marbled Skipper (Gomalia elma)","Tailed Jay (Graphium agamemnon)","Five-bar Swordtail (Graphium antiphates)","Common Jay (Graphium doson)","Spot Swordtail (Graphium nomius)","Southern Bluebottle (Graphium teredon)","South Indian Ace (Halpe hindu)","Moore's Ace (Halpe porus)","Common Awl (Hasora badra)","Common Banded Awl (Hasora chromus)","White Banded Awl (Hasora taminatus)","Plain Banded Awl (Hasora vitta)","Great Orange-tip (Hebomoia glaucippe)","Palni Bushbrown (Heteropsis davisoni/Telinga davisoni)","Common Onyx (Horaga onyx)","Tree Flitter (Hyarotis adrastus)","Great Eggfly (Hypolimnas bolina)","Danaid Eggfly (Hypolimnas misippus)","Nilgiri Tit (Hypolycaena nilgirica)","Chestnut Bob (Iambrix salsala)","Malabar Tree Nymph (Idea malabarica)","Pointed Lineblue (Ionolyce helicon)","Silverstreak Blue (Iraota timoleon)","White Orange-tip (Ixias marianne)","Yellow Orange-tip (Ixias pyrene)","Metallic Cerulean (Jamides alecto)","Dark Cerulean (Jamides bochus)","Common Cerulean (Jamides celeno)","Peacock Pansy (Junonia almana)","Grey Pansy (Junonia atlites)","Yellow Pansy (Junonia hierta)","Chocolate Pansy (Junonia iphita)","Lemon Pansy (Junonia lemonias)","Blue Pansy (Junonia orithiya)","Blue Oakleaf (Kallima horsfieldi)","Blue Admiral (Kaniska canace)","Pea Blue (Lampides boeticus)","Yellow Jack Sailer (Lasippa viraja)","Psyche (Leptosia nina)","Zebra Blue (Leptotes plinius)","Tamil Treebrown (Lethe drypetis)","Bamboo Treebrown (Lethe europa)","Common Treebrown (Lethe rohria)","Southern Beak (Libythea laius)","Club Beak (Libythea myrrha)","Yamfly (Loxura atymnus)","Common Redeye (Matapa aria)","Malayan (Megisba malaya)","Common Evening Brown (Melanitis leda)","Dark Evening Brown (Melanitis phedima)","Great Evening Brown (Melanitis zitenius)","Commander (Moduza procris)","White-bar Bushbrown (Mycalesis anaxias)","Dark-brand Bushbrown (Mycalesis mineus)","Glad-eye Bushbrown (Mycalesis patnia)","Common Bushbrown (Mycalesis perseus)","Tamil Bushbrown (Mycalesis subdita)","Long-brand Bushbrown (Mycalesis visala)","Opaque 6-lineblue (Nacaduba beroe)","Pale 4-lineblue (Nacaduba hermus)","Transparent 6-lineblue (Nacaduba kurava)","Large 4-lineblue (Nacaduba pactolus)","Quaker (Neopithecops zalmora)","Palni Sailer (Neptis (soma)"," palnica)","Sullied Sailer (Neptis clinia)","Common Sailer (Neptis hylas)","Chestnut-streaked Sailer (Neptis jumbah)","Clear Sailer (Neptis nata hampsoni)","Restricted Demon (Notocrypta curvifascia)","Common Banded Demon (Notocrypta paralysos)","Chestnut Angle (Odontoptilum angulatum)","Tamil Dartlet (Oriens concinna)","Indian Dartlet (Oriens goloides)","Nigger (Orsotrioena medus)","Common Rose (Pachliopta aristolochiae)","Crimson Rose (Pachliopta hector)","Malabar Rose (Pachliopta pandiyana)","Common Lascar (Pantoporia hordonia)","Extra Lascar (Pantoporia sandaca)","Malabar Banded Peacock (Papilio buddha)","Common Mime (Papilio clytia)","Common Banded Peacock (Papilio crino)","Lime Butterfly (Papilio demoleus)","Malabar Raven (Papilio dravidarum)","Red Helen (Papilio helenus)","Malabar Banded Swallowtail (Papilio liomedon)","Paris Peacock (Papilio paris)","Blue Mormon (Papilio polymnestor)","Common Mormon (Papilio polytes)","Glassy Tiger (Parantica aglea)","Nilgiri Tiger (Parantica nilgiriensis)","Travancore Evening Brown (Parantirrhoea marshalli)","Dark Wanderer (Pareronia ceylanica)","Common Wanderer (Pareronia hippia)","Oriental Straight Swift (Parnara bada)","Clipper (Parthenos sylvia)","Conjoined Swift (Pelopidas conjuncta)","Small Branded Swift (Pelopidas mathias)","Large Branded Swift (Pelopidas subochracea)","Dingy Lineblue (Petrelaea dana)","Short-banded Sailer (Phaedyma columella)","Small Leopard (Phalanta alcippe)","Common Leopard (Phalanta phalantha)","Indian Cabbage White (Pieris canidia)","Contiguous Swift (Polytremis lubricans)","Anomalous Nawab (Polyura agraria/ Charaxes agrarius)","Common Nawab (Polyura athamas /Charaxes bharata)","Indian Dart (Potanthus sp)","Tufted White Royal (Pratapa deva)","Painted Sawtooth (Prioneris sita)","Tailless Lineblue (Prosotas dubiosa)","Common Lineblue (Prosotas nora)","White-tipped Lineblue (Prosotas noreia)","Fulvous Pied Flat (Pseudocoladenia dan)","Pale Grass Blue (Pseudozizeeria maha)","Coon (Psolos fuligo)","Golden Flitter (Quedara basiflava)","Banded Royal (Rachana jalindra)","Indian Red Flash (Rapala iarbus)","Malabar Flash (Rapala lankana)","Slate Flash (Rapala manea)","Indigo Flash (Rapala varuna)","Monkey Puzzle (Rathinda amor)","Black Prince (Rohana parisatis)","Maculate Lancer (Salanoemia sala)","Common Small Flat (Sarangesa dasahara)","Spotted Small Flat (Sarangesa purendra)","Bicolour Ace (Sovia hyrtacus)","Apefly (Spalgis epius)","Indian Skipper (Spialia galba)","Common Shot Silverline (Spindasis ictis)","Long-banded Silverline (Spindasis lohita)","Plumbeous Silverline (Spindasis schistacea)","Common Silverline (Spindasis vulcanus)","Indian Palm Bob (Suastus gremius)","Common Acacia Blue (Surendra quercetorum)","Baronet (Symphaedra nais)","Suffused Snow Flat (Tagiades gana)","Common Snow Flat (Tagiades japetus)","Water Snow Flat (Tagiades litigiosa)","Peacock Royal (Tajuria cippus)","Plains Blue Royal (Tajuria jehana)","Red Pierrot (Talicada nyseus)","Angled Flat (Tapena thwaitesi)","Tamil Grass Dart (Taractrocera ceramas)","Common Grass Dart (Taractrocera maevius)","Dark Pierrot (Tarucus ananda)","Rounded Pierrot (Tarucus nara)","Dark Palm-Dart (Telicota (ancilla)"," bambusae)","Pale Palm-Dart (Telicota colon)","Red-eye Bushbrown (Telinga adolphei)","Red-disc Bushbrown (Telinga oculus)","Many-tailed Oakblue (Thaduka multicaudata)","Southern Spotted Ace (Thoressa astigmata)","Evershed's Ace (Thoressa evershedi)","Madras Ace (Thoressa honorei)","Blue Tiger (Tirumala limniace)","Dark Blue Tiger (Tirumala septentrionis)","Southern Birdwing (Troides minos)","	White Hedge Blue (Udara akasa)","Grass Demon (Udaspes folus)","Painted lady (Vanessa cardui)","Indian Red Admiral (Vanessa indica)","Cruiser (Vindula erota)","Guava Blue (Virachola isocrates)","Large Guava Blue (Virachola perse)","Common Three-ring (Ypthima asterope)","Common Five-ring (Ypthima baldus)","White Four-ring (Ypthima ceylonica)","Nilgiri Four-ring (Ypthima chenu)","Common Four-ring (Ypthima huebneri)","Nilgiri Jewel Four-ring (Ypthima striata)","Palni Four-ring (Ypthima ypthimoides)","Fluffy Tit (Zeltus amasa)","Redspot (Zesius chrysomallus)","Silver-streaked Acacia Blue (Zinaspa todara)","Tamil Catseye (Zipaetis saitis)","Dark Grass Blue (Zizeeria karsandra)","Lesser Grass Blue (Zizina otis)","Tiny Grass Blue (Zizula hylax)","Purple Spotted Flitter (Zographetus ogygia)"]
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
