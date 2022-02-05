def format_symbol_for_api(symbol: str):
    symbol = symbol.upper()
    symbol = symbol.replace("/", "")
    symbol = symbol.split(":")[-1]

    currencies = [
        "AUDCAD",
        "AUDCHF",
        "AUDCZK",
        "AUDDKK",
        "AUDHKD",
        "AUDHUF",
        "AUDJPY",
        "AUDMXN",
        "AUDNOK",
        "AUDNZD",
        "AUDPLN",
        "AUDSEK",
        "AUDSGD",
        "AUDUSD",
        "AUDZAR",
        "CADCHF",
        "CADCZK",
        "CADDKK",
        "CADHKD",
        "CADHUF",
        "CADJPY",
        "CADMXN",
        "CADNOK",
        "CADPLN",
        "CADSEK",
        "CADSGD",
        "CADZAR",
        "CHFCZK",
        "CHFDKK",
        "CHFHKD",
        "CHFHUF",
        "CHFJPY",
        "CHFMXN",
        "CHFNOK",
        "CHFPLN",
        "CHFSEK",
        "CHFSGD",
        "CHFTRY",
        "CHFZAR",
        "DKKCZK",
        "DKKHKD",
        "DKKHUF",
        "DKKMXN",
        "DKKNOK",
        "DKKPLN",
        "DKKSEK",
        "DKKSGD",
        "DKKZAR",
        "EURAUD",
        "EURCAD",
        "EURCHF",
        "EURCZK",
        "EURDKK",
        "EURGBP",
        "EURHKD",
        "EURHUF",
        "EURJPY",
        "EURMXN",
        "EURNOK",
        "EURNZD",
        "EURPLN",
        "EURSEK",
        "EURSGD",
        "EURTRY",
        "EURUSD",
        "EURZAR",
        "GBPAUD",
        "GBPCAD",
        "GBPCHF",
        "GBPCZK",
        "GBPDKK",
        "GBPHKD",
        "GBPHUF",
        "GBPJPY",
        "GBPMXN",
        "GBPNOK",
        "GBPNZD",
        "GBPPLN",
        "GBPSEK",
        "GBPSGD",
        "GBPUSD",
        "GBPZAR",
        "JPYCZK",
        "JPYDKK",
        "JPYHKD",
        "JPYHUF",
        "JPYMXN",
        "JPYNOK",
        "JPYPLN",
        "JPYSEK",
        "JPYSGD",
        "JPYZAR",
        "NOKCZK",
        "NOKHKD",
        "NOKHUF",
        "NOKMXN",
        "NOKPLN",
        "NOKSEK",
        "NOKSGD",
        "NOKZAR",
        "NZDCAD",
        "NZDCHF",
        "NZDCZK",
        "NZDDKK",
        "NZDHKD",
        "NZDHUF",
        "NZDJPY",
        "NZDMXN",
        "NZDNOK",
        "NZDPLN",
        "NZDSEK",
        "NZDSGD",
        "NZDUSD",
        "NZDZAR",
        "USDCAD",
        "USDCHF",
        "USDCZK",
        "USDDKK",
        "USDHKD",
        "USDHUF",
        "USDJPY",
        "USDMXN",
        "USDNOK",
        "USDPLN",
        "USDSEK",
        "USDSGD",
        "USDTRY",
        "USDZAR"
    ]

    cryptocurrencies = [
        "BTCUSDT",
        "ETHUSDT",
        "BUSDUSDT",
        "FTMUSDT",
        "ETHBUSD",
        "BNBUSDT",
        "BTCBUSD",
        "SANDUSDT",
        "ATOMUSDT",
        "MATICUSDT",
        "LUNAUSDT",
        "AVAXUSDT",
        "XRPUSDT",
        "CRVUSDT",
        "DOTUSDT",
        "ROSEUSDT",
        "ADAUSDT",
        "SOLUSDT",
        "ETHBTC",
        "TRXUSDT",
        "SHIBUSDT",
        "GALAUSDT",
        "NEARUSDT",
        "SUSHIUSDT",
        "SXPUSDT",
        "FARMUSDT",
        "LINKUSDT",
        "ANTUSDT",
        "MANAUSDT",
        "LUNABUSD",
        "USDTTRY",
        "USDCUSDT",
        "FILUSDT",
        "BTCUSDC",
        "DOGEUSDT",
        "BNBBUSD",
        "CHRUSDT",
        "KP3RUSDT",
        "LUNAEUR",
        "ICPUSDT",
        "KAVAUSDT",
        "BTCEUR",
        "YFIUSDT",
        "ROSEBUSD",
        "AXSUSDT",
        "MATICBUSD",
        "ALGOUSDT",
        "BNBBTC",
        "ETHEUR",
        "ONEUSDT",
        "JASMYUSDT",
        "LTCUSDT",
        "THETAUSDT",
        "LRCUSDT",
        "AVAXBUSD",
        "FXSUSDT",
        "JSTUSDT",
        "VOXELUSDT",
        "LUNABTC",
        "WINUSDT",
        "AAVEUSDT",
        "OMGUSDT",
        "MATICBTC",
        "FTMBUSD",
        "SHIBBUSD",
        "CVXUSDT",
        "EOSUSDT",
        "SOLBUSD",
        "UNIUSDT",
        "GXSUSDT",
        "VETUSDT",
        "ENJUSDT",
        "KEEPUSDT",
        "XTZUSDT",
        "TUSDUSDT",
        "ALICEUSDT",
        "USDCBUSD",
        "BTTUSDT",
        "ARUSDT",
        "IOTAUSDT",
        "DYDXUSDT",
        "TLMUSDT",
        "KP3RBUSD",
        "FUNUSDT",
        "AVAXTRY",
        "RVNUSDT",
        "SANDBUSD",
        "XMRUSDT",
        "ADABTC",
        "SHIBTRY",
        "ETHUSDC",
        "FTTUSDT",
        "EGLDUSDT",
        "DOTBTC",
        "EURUSDT",
        "PEOPLEUSDT",
        "ADABUSD",
        "MASKUSDT",
        "ATOMBTC",
        "DUSKUSDT",
        "BCHUSDT",
        "FARMBUSD",
        "CELOUSDT",
        "ATOMBUSD",
        "XRPBTC",
        "FTMBTC",
        "AVAXBTC",
        "CHZUSDT",
        "GTCUSDT",
        "DOTBUSD",
        "ETCUSDT",
        "SOLBTC",
        "GRTUSDT",
        "ENSUSDT",
        "BATUSDT",
        "XRPBUSD",
        "ZECUSDT",
        "LINAUSDT",
        "XLMUSDT",
        "CTXCUSDT",
        "ORNUSDT",
        "CRVBUSD",
        "1INCHUSDT",
        "MDTUSDT",
        "FETUSDT",
        "SLPUSDT",
        "LTOUSDT",
        "BUSDTRY",
        "MBOXUSDT",
        "CAKEUSDT",
        "ZILUSDT",
        "HNTUSDT",
        "XMRBTC",
        "QUICKUSDT",
        "OOKIUSDT",
        "FARMBTC",
        "DARUSDT",
        "AUDUSDT",
        "FXSBUSD",
        "BNBETH",
        "SXPEUR",
        "HOTUSDT",
        "DREPUSDT",
        "FLUXUSDT",
        "ICXUSDT",
        "USTUSDT",
        "HBARUSDT",
        "ROSEBTC",
        "CELRUSDT",
        "DASHUSDT",
        "LUNABNB",
        "VOXELBUSD",
        "MANABUSD",
        "SYSUSDT",
        "USDTBRL",
        "ALCXUSDT",
        "LINKBTC",
        "CRVBTC",
        "DOGEBUSD",
        "LTCBTC",
        "SNXUSDT",
        "FORUSDT",
        "SANDBTC",
        "RUNEUSDT",
        "ETHDAI",
        "AUDIOUSDT",
        "DENTUSDT",
        "SPELLUSDT",
        "ONEBUSD",
        "ANTBUSD",
        "COTIUSDT",
        "NEARBTC",
        "WBTCBTC",
        "IOTXUSDT",
        "C98USDT",
        "KLAYUSDT",
        "NEARBUSD",
        "BTCTRY",
        "SFPUSDT",
        "IOSTUSDT",
        "LUNAETH",
        "JASMYBUSD",
        "HIGHUSDT",
        "SXPTRY",
        "USDTDAI",
        "FLOWUSDT",
        "GALABUSD",
        "MANABTC",
        "MINAUSDT",
        "DOTUPUSDT",
        "BTCGBP",
        "SUSHIBTC",
        "CHESSUSDT",
        "YFIBUSD",
        "MANATRY",
        "BICOUSDT",
        "AAVEBTC",
        "SRMUSDT",
        "BTCTUSD",
        "UNIBTC",
        "YFIBTC",
        "TLMTRY",
        "GBPUSDT",
        "EURBUSD",
        "NEOUSDT",
        "PHAUSDT",
        "RSRUSDT",
        "AXSBUSD",
        "ICPBUSD",
        "LINKBUSD",
        "ETHAUD",
        "REEFUSDT",
        "ETHGBP",
        "ONTUSDT",
        "RENUSDT",
        "XMRETH",
        "WAVESUSDT",
        "ADAEUR",
        "VOXELBTC",
        "OCEANUSDT",
        "ANKRUSDT",
        "RNDRUSDT",
        "BTCAUD",
        "XRPEUR",
        "ETHTRY",
        "COCOSUSDT",
        "SUSHIBUSD",
        "WAXPUSDT",
        "COSUSDT",
        "KEYUSDT",
        "BUSDBRL",
        "SANDTRY",
        "OGNUSDT",
        "SXPBTC",
        "HOTTRY",
        "USTBUSD",
        "COMPUSDT",
        "PYRUSDT",
        "LRCBTC",
        "SUNUSDT",
        "BTCBRL",
        "ANTBTC",
        "VETBUSD",
        "BAKEUSDT",
        "DODOUSDT",
        "UNFIUSDT",
        "CVCUSDT",
        "FXSBTC",
        "YGGUSDT",
        "BLZUSDT",
        "QTUMUSDT",
        "CVXBUSD",
        "JOEUSDT",
        "CHRBUSD",
        "BTCDAI",
        "CTSIUSDT",
        "ONEBTC",
        "DOGEBTC",
        "BNBEUR",
        "MATICTRY",
        "FTTBUSD",
        "CVXBTC",
        "LRCBUSD",
        "SXPBUSD",
        "RVNBTC",
        "SYSBUSD",
        "CAKEBUSD",
        "PROSETH",
        "AXSBTC",
        "STORJUSDT",
        "BCHBTC",
        "DARBUSD",
        "BTCDOWNUSDT",
        "KSMUSDT",
        "TFUELUSDT",
        "SUPERUSDT",
        "XLMBTC",
        "ALGOBUSD",
        "ARPATRY",
        "ENJBTC",
        "USDTBIDR",
        "REQUSDT",
        "JASMYBTC",
        "BNXUSDT",
        "YFIIUSDT",
        "LTOBTC",
        "ALGOBTC",
        "BATBTC",
        "BNBBRL",
        "TLMBUSD",
        "BONDUSDT",
        "SHIBEUR",
        "HBARBTC",
        "LPTUSDT",
        "DOTBNB",
        "ILVUSDT",
        "OOKIBUSD",
        "ZENUSDT",
        "USDTRUB",
        "RVNTRY",
        "MATICBNB",
        "XTZUPUSDT",
        "FILBUSD",
        "SKLUSDT",
        "QNTUSDT",
        "EPSUSDT",
        "NUUSDT",
        "ATAUSDT",
        "LITUSDT",
        "CAKEBTC",
        "ZRXUSDT",
        "EGLDBUSD",
        "LAZIOUSDT",
        "KEEPBTC",
        "ENJBUSD",
        "MDXUSDT",
        "HARDUSDT",
        "XEMUSDT",
        "SCUSDT",
        "MATICEUR",
        "CTKUSDT",
        "LTCBUSD",
        "AVAXETH",
        "XTZBTC",
        "BETAUSDT",
        "DOGETRY",
        "WRXUSDT",
        "FLUXBUSD",
        "BTCUPUSDT",
        "XECUSDT",
        "STXUSDT",
        "ETHBRL",
        "AVAXBNB",
        "MITHUSDT",
        "MLNUSDT",
        "NULSUSDT",
        "DGBUSDT",
        "AUTOUSDT",
        "THETABTC",
        "ARPAUSDT",
        "SOLTRY",
        "QUICKBUSD",
        "XTZBUSD",
        "ADAETH",
        "GXSBTC",
        "BALUSDT",
        "TCTUSDT",
        "FETBTC",
        "GALABTC",
        "MATICETH",
        "FTTBTC",
        "ADATRY",
        "SOLETH",
        "MKRUSDT",
        "IOTABUSD",
        "CHRBTC",
        "KAVABTC",
        "BUSDDAI",
        "KP3RBNB",
        "ALPACAUSDT",
        "DOTDOWNUSDT",
        "MCUSDT",
        "BELUSDT",
        "XRPUSDC",
        "ALICEBUSD",
        "DUSKBTC",
        "CKBUSDT",
        "VETBTC",
        "DYDXBUSD",
        "RLCUSDT",
        "SOLBNB",
        "EOSBUSD",
        "QIUSDT",
        "FTMTRY",
        "TRXBTC",
        "AAVEBUSD",
        "LAZIOTRY",
        "TRXBUSD",
        "ETHTUSD",
        "LINKETH",
        "AUDBUSD",
        "PEOPLEBUSD",
        "GTOUSDT",
        "SPARTABNB",
        "XRPBNB",
        "FORBUSD",
        "SOLEUR",
        "BTTUSDC",
        "BTTTUSD",
        "RAYUSDT",
        "SOLUSDC",
        "FILBTC",
        "PONDUSDT",
        "GBPBUSD",
        "PAXGUSDT",
        "FTMETH",
        "BANDUSDT",
        "BTTTRY",
        "ORNBUSD",
        "MBOXBUSD",
        "IDEXUSDT",
        "ENSBUSD",
        "ADXUSDT",
        "OGUSDT",
        "DOTTRY",
        "TVKUSDT",
        "ICPBTC",
        "ARBTC",
        "DOTEUR",
        "FARMBNB",
        "KNCUSDT",
        "XRPETH",
        "SYSBTC",
        "ARBUSD",
        "UNIBUSD",
        "KEEPBUSD",
        "SLPBUSD",
        "STMXUSDT",
        "XTZDOWNUSDT",
        "BNBTRY",
        "MOVRUSDT",
        "REEFTRY",
        "CHZTRY",
        "XRPTRY",
        "COMPBUSD",
        "AVAXEUR",
        "KAVABUSD",
        "FUNETH",
        "ORNBTC",
        "ADAUSDC",
        "LUNATRY",
        "CELOBUSD",
        "ZECBTC",
        "PYRBTC",
        "TRUUSDT",
        "UMAUSDT",
        "ALPHAUSDT",
        "RUNEBUSD",
        "ALCXBUSD",
        "IOTABTC",
        "XLMBUSD",
        "AGLDUSDT",
        "ICXBTC",
        "CHZBTC",
        "MASKBUSD",
        "BTCBIDR",
        "BNBUSDC",
        "ZILBTC",
        "AXSETH",
        "IOSTBTC",
        "ETHUPUSDT",
        "AKROUSDT",
        "FLUXBTC",
        "MIRUSDT",
        "ETHDOWNUSDT",
        "ATOMUSDC",
        "CLVUSDT",
        "SCRTBUSD",
        "AUCTIONUSDT",
        "PNTUSDT",
        "ADABNB",
        "HNTBUSD",
        "HBARBUSD",
        "ALICEBTC",
        "LINKDOWNUSDT",
        "WTCUSDT",
        "WABIBTC",
        "GNOUSDT",
        "TFUELBTC",
        "RAREUSDT",
        "HNTBTC",
        "WAXPBUSD",
        "VGXUSDT",
        "QUICKBTC",
        "FORBTC",
        "BICOBUSD",
        "VTHOUSDT",
        "XVSUSDT",
        "BTTTRX",
        "ADXETH",
        "SUSHIBNB",
        "LRCTRY",
        "TUSDBUSD",
        "SNXBTC",
        "NANOUSDT",
        "BUSDBIDR",
        "XRPUPUSDT",
        "SLPETH",
        "TWTUSDT",
        "XRPDOWNUSDT",
        "REPUSDT",
        "ADADOWNUSDT",
        "AMPUSDT",
        "HIVEUSDT",
        "GALATRY",
        "TLMBTC",
        "CHESSBUSD",
        "BTCSTUSDT",
        "LINKUPUSDT",
        "VIDTUSDT"
    ]

    if symbol in currencies:
        symbol = f"C:{symbol}"
    elif symbol in cryptocurrencies:
        symbol = f"X:{symbol}"

    return symbol