import json
import logging
import os

import torch
from torch.nn.utils import weight_norm

from clip import clip
from typing import Sequence, Optional
from itertools import islice


TEMPLATES = {
    'EuroSAT': 
        ['a centered satellite photo of {}.', 
         'a centered satellite photo of a {}.', 
         'a centered satellite photo of the {}.',],
    'Aircraft':
        ['a photo of a {}, a type of aircraft.',
        'a photo of the {}, a type of aircraft.',],
    'Caltech101':
        ['a photo of a {}.',
        'a painting of a {}.',
        'a plastic {}.',
        'a sculpture of a {}.',
        'a sketch of a {}.',
        'a tattoo of a {}.',
        'a toy {}.',
        'a rendition of a {}.',
        'a embroidered {}.',
        'a cartoon {}.',
        'a {} in a video game.',
        'a plushie {}.',
        'a origami {}.',
        'art of a {}.',
        'graffiti of a {}.',
        'a drawing of a {}.',
        'a doodle of a {}.',
        'a photo of the {}.',
        'a painting of the {}.',
        'the plastic {}.',
        'a sculpture of the {}.',
        'a sketch of the {}.',
        'a tattoo of the {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'the embroidered {}.',
        'the cartoon {}.',
        'the {} in a video game.',
        'the plushie {}.',
        'the origami {}.',
        'art of the {}.',
        'graffiti of the {}.',
        'a drawing of the {}.',
        'a doodle of the {}.',],
    "Cars":
        ['a photo of a {}.',
        'a photo of the {}.',
        'a photo of my {}.',
        'i love my {}!',
        'a photo of my dirty {}.',
        'a photo of my clean {}.',
        'a photo of my new {}.',
        'a photo of my old {}.',],
    "Food101":
        ['a photo of {}, a type of food.',],
    "Pets":
        ['a photo of a {}, a type of pet.',],
    "Flowers":
        ['a photo of a {}, a type of flower.',],
    "DTD":
        ['a photo of a {} texture.',
        'a photo of a {} pattern.',
        'a photo of a {} thing.',
        'a photo of a {} object.',
        'a photo of the {} texture.',
        'a photo of the {} pattern.',
        'a photo of the {} thing.',
        'a photo of the {} object.',],
    "UCF101":
        ['a photo of a person {}.',
        'a video of a person {}.',
        'a example of a person {}.',
        'a demonstration of a person {}.',
        'a photo of the person {}.',
        'a video of the person {}.',
        'a example of the person {}.',
        'a demonstration of the person {}.',
        'a photo of a person using {}.',
        'a video of a person using {}.',
        'a example of a person using {}.',
        'a demonstration of a person using {}.',
        'a photo of the person using {}.',
        'a video of the person using {}.',
        'a example of the person using {}.',
        'a demonstration of the person using {}.',
        'a photo of a person doing {}.',
        'a video of a person doing {}.',
        'a example of a person doing {}.',
        'a demonstration of a person doing {}.',
        'a photo of the person doing {}.',
        'a video of the person doing {}.',
        'a example of the person doing {}.',
        'a demonstration of the person doing {}.',
        'a photo of a person during {}.',
        'a video of a person during {}.',
        'a example of a person during {}.',
        'a demonstration of a person during {}.',
        'a photo of the person during {}.',
        'a video of the person during {}.',
        'a example of the person during {}.',
        'a demonstration of the person during {}.',
        'a photo of a person performing {}.',
        'a video of a person performing {}.',
        'a example of a person performing {}.',
        'a demonstration of a person performing {}.',
        'a photo of the person performing {}.',
        'a video of the person performing {}.',
        'a example of the person performing {}.',
        'a demonstration of the person performing {}.',
        'a photo of a person practicing {}.',
        'a video of a person practicing {}.',
        'a example of a person practicing {}.',
        'a demonstration of a person practicing {}.',
        'a photo of the person practicing {}.',
        'a video of the person practicing {}.',
        'a example of the person practicing {}.',
        'a demonstration of the person practicing {}.',],
    "SUN397":
        ['a photo of a {}.',
        'a photo of the {}.',],
    "ImageNet":
        ['itap of a {}.',
        'a bad photo of the {}.',
        'a origami {}.',
        'a photo of the large {}.',
        'a {} in a video game.',
        'art of the {}.',
        'a photo of the small {}.']
         
}


CLASSNAME_MAP = {
    "flowers": {
        "air plant": "ball moss",
        "globe flower": "globe-flower",
        "pink and yellow dahlia": "pink-yellow dahlia"
    },
    "eurosat": {
        "forest": "Forest",
        "permanent crop land": "Permanent Crop Land",
        "residential buildings or homes or apartments": "Residential Buildings",
        "river": "River",
        "pasture land": "Pasture Land",
        "lake or sea": "Sea or Lake",
        "brushland or shrubland": "Herbaceous Vegetation Land",
        "annual crop land": "Annual Crop Land",
        "industrial buildings or commercial buildings": "Industrial Buildings",
        "highway or road": "Highway or Road"
    },
    "sun397": {
        "car interior backseat": "backseat car interior",
        "wine cellar barrel storage": "barrel storage wine cellar",
        "stadium baseball": "baseball stadium",
        "waterfall block": "block waterfall",
        "wine cellar bottle storage": "bottle storage wine cellar",
        "forest broadleaf": "broadleaf forest",
        "underwater coral reef": "coral reef underwater",
        "field cultivated": "cultivated field",
        "elevator door": "door elevator",
        "temple east asia": "east asia temple",
        "poolroom establishment": "establishment poolroom",
        "balcony exterior": "exterior balcony",
        "covered bridge exterior": "exterior covered bridge",
        "gazebo exterior": "exterior gazebo",
        "waterfall fan": "fan waterfall",
        "stadium football": "football stadium",
        "car interior frontseat": "frontseat car interior",
        "dinette home": "home dinette",
        "poolroom home": "home poolroom",
        "apse indoor": "indoor apse",
        "badminton court indoor": "indoor badminton court",
        "bazaar indoor": "indoor bazaar",
        "bistro indoor": "indoor bistro",
        "booth indoor": "indoor booth",
        "bow window indoor": "indoor bow window",
        "brewery indoor": "indoor brewery",
        "casino indoor": "indoor casino",
        "cathedral indoor": "indoor cathedral",
        "cavern indoor": "indoor cavern",
        "chicken coop indoor": "indoor chicken coop",
        "church indoor": "indoor church",
        "cloister indoor": "indoor cloister",
        "diner indoor": "indoor diner",
        "escalator indoor": "indoor escalator",
        "factory indoor": "indoor factory",
        "firing range indoor": "indoor firing range",
        "florist shop indoor": "indoor florist shop",
        "garage indoor": "indoor garage",
        "general store indoor": "indoor general store",
        "greenhouse indoor": "indoor greenhouse",
        "gymnasium indoor": "indoor gymnasium",
        "hangar indoor": "indoor hangar",
        "ice skating rink indoor": "indoor ice skating rink",
        "jacuzzi indoor": "indoor jacuzzi",
        "jail indoor": "indoor jail",
        "kennel indoor": "indoor kennel",
        "library indoor": "indoor library",
        "market indoor": "indoor market",
        "mosque indoor": "indoor mosque",
        "movie theater indoor": "indoor movie theater",
        "museum indoor": "indoor museum",
        "parking garage indoor": "indoor parking garage",
        "pilothouse indoor": "indoor pilothouse",
        "podium indoor": "indoor podium",
        "theater indoor procenium": "indoor procenium theater",
        "pub indoor": "indoor pub",
        "theater indoor seats": "indoor seats theater",
        "shopping mall indoor": "indoor shopping mall",
        "stage indoor": "indoor stage",
        "swimming pool indoor": "indoor swimming pool",
        "synagogue indoor": "indoor synagogue",
        "tennis court indoor": "indoor tennis court",
        "volleyball court indoor": "indoor volleyball court",
        "warehouse indoor": "indoor warehouse",
        "wrestling ring indoor": "indoor wrestling ring",
        "balcony interior": "interior balcony",
        "elevator interior": "interior elevator",
        "canal natural": "natural canal",
        "lake natural": "natural lake",
        "forest needleleaf": "needleleaf forest",
        "cubicle office": "office cubicle",
        "apartment building outdoor": "outdoor apartment building",
        "arrival gate outdoor": "outdoor arrival gate",
        "athletic field outdoor": "outdoor athletic field",
        "basketball court outdoor": "outdoor basketball court",
        "bazaar outdoor": "outdoor bazaar",
        "bow window outdoor": "outdoor bow window",
        "cabin outdoor": "outdoor cabin",
        "cathedral outdoor": "outdoor cathedral",
        "chicken coop outdoor": "outdoor chicken coop",
        "church outdoor": "outdoor church",
        "control tower outdoor": "outdoor control tower",
        "diner outdoor": "outdoor diner",
        "doorway outdoor": "outdoor doorway",
        "driving range outdoor": "outdoor driving range",
        "general store outdoor": "outdoor general store",
        "greenhouse outdoor": "outdoor greenhouse",
        "hangar outdoor": "outdoor hangar",
        "hot tub outdoor": "outdoor hot tub",
        "hotel outdoor": "outdoor hotel",
        "hunting lodge outdoor": "outdoor hunting lodge",
        "ice skating rink outdoor": "outdoor ice skating rink",
        "inn outdoor": "outdoor inn",
        "kennel outdoor": "outdoor kennel",
        "labyrinth outdoor": "outdoor labyrinth",
        "library outdoor": "outdoor library",
        "lido deck outdoor": "outdoor lido deck",
        "market outdoor": "outdoor market",
        "monastery outdoor": "outdoor monastery",
        "mosque outdoor": "outdoor mosque",
        "nuclear power plant outdoor": "outdoor nuclear power plant",
        "observatory outdoor": "outdoor observatory",
        "oil refinery outdoor": "outdoor oil refinery",
        "outhouse outdoor": "outdoor outhouse",
        "parking garage outdoor": "outdoor parking garage",
        "planetarium outdoor": "outdoor planetarium",
        "podium outdoor": "outdoor podium",
        "power plant outdoor": "outdoor power plant",
        "swimming pool outdoor": "outdoor swimming pool",
        "synagogue outdoor": "outdoor synagogue",
        "tennis court outdoor": "outdoor tennis court",
        "tent outdoor": "outdoor tent",
        "track outdoor": "outdoor track",
        "volleyball court outdoor": "outdoor volleyball court",
        "subway station platform": "platform subway station",
        "train station platform": "platform train station",
        "waterfall plunge": "plunge waterfall",
        "atrium public": "public atrium",
        "desert sand": "sand desert",
        "bakery shop": "shop bakery",
        "temple south asia": "south asia temple",
        "canal urban": "urban canal",
        "desert vegetation": "vegetation desert",
        "dinette vehicle": "vehicle dinette",
        "moat water": "water moat",
        "field wild": "wild field"
    }
}

import torch.nn as nn


class ImageClassifierWithCLIP:
    def __init__(self, args, classnames, device='cpu', text_adapter=None):
        self.device = device
        self.category = args.category
        self.model, _ = clip.load(args.clip_path, device=self.device)
        
        for p in self.model.parameters():
            p.requires_grad = False
        self.classnames = [classname.replace('_', '/') for classname in classnames]

        self.template_type = args.template_type
        
        if self.category == 'imagenet':
            self.classnames = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                        "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                        "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                        "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
                        "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
                        "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
                        "box turtle", "banded gecko", "green iguana", "Carolina anole",
                        "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
                        "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
                        "American alligator", "triceratops", "worm snake", "ring-necked snake",
                        "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
                        "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
                        "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
                        "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
                        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
                        "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
                        "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
                        "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
                        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
                        "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
                        "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
                        "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
                        "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
                        "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
                        "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
                        "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
                        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
                        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
                        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
                        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
                        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
                        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
                        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
                        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
                        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
                        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
                        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
                        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
                        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
                        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
                        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
                        "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                        "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                        "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                        "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
                        "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
                        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
                        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
                        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
                        "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
                        "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
                        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
                        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
                        "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
                        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
                        "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
                        "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
                        "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
                        "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
                        "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
                        "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
                        "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
                        "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
                        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
                        "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
                        "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
                        "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
                        "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
                        "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
                        "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
                        "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
                        "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
                        "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
                        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
                        "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
                        "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
                        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
                        "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
                        "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
                        "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
                        "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
                        "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
                        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
                        "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
                        "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
                        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
                        "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
                        "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
                        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
                        "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
                        "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
                        "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
                        "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
                        "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
                        "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
                        "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
                        "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
                        "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
                        "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
                        "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
                        "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
                        "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
                        "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
                        "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
                        "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
                        "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
                        "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
                        "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
                        "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
                        "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
                        "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
                        "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
                        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
                        "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
                        "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
                        "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
                        "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
                        "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
                        "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
                        "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
                        "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
                        "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
                        "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
                        "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
                        "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
                        "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
                        "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
                        "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
                        "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
                        "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
                        "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
                        "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
                        "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
                        "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
                        "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
                        "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
                        "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
                        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
                        "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
                        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
                        "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
                        "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
                        "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
                        "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
                        "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                        "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                        "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                        "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                        "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                        "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                        "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
                        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
                        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
                        "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                        "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

        self.templates = TEMPLATES['ImageNet']

        self.zeroshot_weights = self.build_zero_shot_classifier()


    def build_zero_shot_classifier(
            self,
            num_classes_per_batch: Optional[int] = 64
    ):
        """Build zero-shot classifier weights by iterating over class names in batches."""
        assert isinstance(self.templates, Sequence) and len(self.templates) > 0
        assert isinstance(self.classnames, Sequence) and len(self.classnames) > 0
        use_format = isinstance(self.templates[0], str)
        num_templates = len(self.templates)
        num_classes = len(self.classnames)

        def _process_batch(batch_classnames):
            """
            Process 80 templates as batches
            """
            num_batch_classes = len(batch_classnames)
            texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in self.templates]
            texts = clip.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
            class_embeddings = class_embeddings.T
            return class_embeddings

        def _cafo_weights(classnames, iters, use_both=True):
            PATH_TO_PROMPTS = f'dinov2/eval/prompt/{self.category}_prompts.json'
            with open(PATH_TO_PROMPTS) as f:
                gpt3_prompts = json.load(f)

            with torch.no_grad():
                zeroshot_weights = []
                i = 0
                for idx, classname in enumerate(classnames):
                    classname = classname.replace('_', '/')
                    
                    # Check the name is consist with CaFo name ??
                    replace = False
                    if self.category.lower() in CLASSNAME_MAP.keys():
                        if classname in CLASSNAME_MAP[self.category.lower()].keys():
                            classname_ = CLASSNAME_MAP[self.category.lower()][classname]
                            replace = True
                        else:
                            classname_ = classname
                    else:
                        classname_ = classname
                    
                    if use_both:
                        # Hand craft with names
                        texts = [template.format(classname) for template in self.templates]
                    else:
                        texts = []
                    
                    # replace names CaFo prompt to OpenAI names
                    for t in gpt3_prompts[classname_]:
                        if replace:
                            print(f"Replacing {classname_} with {classname}")
                            t = t.replace(classname_, classname)
                        texts.append(t)

                    texts = clip.tokenize(texts, truncate=True).cuda()  # tokenize
                    class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                    i += 1
                zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

            return zeroshot_weights

        if 'cafo' in self.template_type:
            use_both = False
            if 'simple' in self.template_type or 'base' in self.template_type:
                use_both = True
            zeroshot_weights = _cafo_weights(self.classnames, use_both)
        else:
            if num_classes_per_batch:
                batched_embeds = [_process_batch(batch) for batch in
                                  self.batched(self.template_classnames, num_classes_per_batch)]
                zeroshot_weights = torch.cat(batched_embeds, dim=1)
            else:
                zeroshot_weights = _process_batch(self.template_classnames)

        return zeroshot_weights

    def batched(self, iterable, n):
        """Batch data into lists of length n. The last batch may be shorter."""
        it = iter(iterable)
        while batch := list(islice(it, n)):
            yield batch

    def predict(self, image_features):
        # Compute similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ self.zeroshot_weights
        return similarity


class ClassificationHead(torch.nn.Module):
    def __init__(self, state_dict, weights, normalize=True, biases=None):
        weights_t = torch.transpose(weights.clone(), 1, 0)
        output_size, input_size = weights_t.shape
        super(ClassificationHead, self).__init__()
        self.normalize = normalize
        self.text_classifier = torch.nn.Linear(input_size, output_size)
        if state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k.startswith("text_classifier")}
            state_dict = {k.replace("text_classifier.", ""): v for k, v in state_dict.items()}
            self.text_classifier.load_state_dict(state_dict)
        else:
            if weights is not None:
                self.text_classifier.weight.data = weights_t.data
            if biases is not None:
                self.bias = torch.nn.Parameter(biases.clone())
            else:
                self.bias = torch.nn.Parameter(torch.zeros_like(self.text_classifier.bias), requires_grad=True)
            self.text_classifier.bias.data = torch.zeros_like(self.text_classifier.bias).data

    def forward(self, inputs):
        if self.normalize:
            norm_inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        else:
            norm_inputs = inputs
        logits = self.text_classifier(norm_inputs)

        return logits


class WNClassificationHead(torch.nn.Module):
    def __init__(self, state_dict, weights, normalize=True):
        weights_t = torch.transpose(weights.clone(), 1, 0)
        output_size, input_size = weights_t.shape
        super(WNClassificationHead, self).__init__()
        self.normalize = normalize
        text_classifier = torch.nn.Linear(input_size, output_size, bias=False)
        if state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k.startswith("text_classifier")}
            state_dict = {k.replace("text_classifier.", ""): v for k, v in state_dict.items()}
            text_classifier.weight.data = state_dict["weight_v"]
        else:
            if weights is not None:
                text_classifier.weight.data = weights_t.data
        self.text_classifier = weight_norm(text_classifier)
        self.text_classifier.weight_g.requires_grad = False

    def forward(self, inputs):
        if self.normalize:
            norm_inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        else:
            norm_inputs = inputs
        logits = self.text_classifier(norm_inputs)

        return logits


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs
