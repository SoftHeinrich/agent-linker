#!/usr/bin/env python3
"""MANUAL re-annotation of the REFERENCE FORM of every gold doc-model trace link.

Population: the 195 gold (sentence, component) links over the five SAD/model
pairs, derived by ``audit.py --sheets DIR`` into ``ch1_gold_sheet.csv``.  That
sheet carries an AUTOMATIC guess (`auto_surface`) produced by contiguous-token
containment; this file records a HUMAN verdict for each of the 195 links and
nothing else.  Every verdict was reached by reading the sentence -- and, where
the sentence carries no name, its neighbours -- in the SAD itself.

What is being tested
--------------------
paper/sections/approach.tex (Challenge 1) claims links are carried by "various
reference forms": a name, a document alias, part of a multi-word name, or a
pronoun.  The question is whether that variety is real and general, or an
artifact of the automatic classifier and of one project's naming.

Categories (exactly one per link, the expression that identifies the component)
------------------------------------------------------------------------------
canonical
    The catalog name itself occurs in the sentence, in a use that is about the
    component.  Orthographic and inflectional variation is still `canonical`:
    case ("webui"/"WebUi"), spacing ("Image Provider" for `ImageProvider`),
    hyphenation ("bbb-web" for `BBB web`), camelCase splitting, plural, and a
    longer compound that CONTAINS the whole name ("PersistenceProvider",
    "UI Server", "akka-apps").  Nothing outside the name has to be known to
    resolve these -- only normalisation.  Tie-break: if the name string occurs
    AND the clause also uses an anaphor, the link is `canonical` (it is
    lexically recoverable); the note records the anaphor.

alias
    The document denotes the component with a DIFFERENT lexical string that
    normalisation alone cannot reach: an expansion of an abbreviation
    ("Database" for `DB`, "BigBlueButton web application" for `BBB web`,
    "FreeSWITCH Event Socket Layer" for `FSESL`), a contraction ("KMS" for
    `kurento`), a deployment/package name ("bbb-html5" for `HTML5 Server`), or
    an outright substitute noun ("DataStorage" for `FileStorage`,
    "AudioAccess" for `MediaAccess`, "ReEncoder" for `Reencoding`).
    Resolving these needs the document's own naming convention.

partial
    Only PART of a multi-word / compound catalog name occurs: a distinctive
    word ("datastore" for `GAE Datastore`, "WebRTC" for `WebRTC-SFU`) or only
    the generic head noun ("client"/"server" for `HTML5 Client`/`HTML5
    Server`, "UI" for `WebUI`).  The rest of the name is absent.

pronoun_or_implicit
    No form of the name at all: a pronoun ("It", "Its", "they"), a
    demonstrative shell ("This component"), or an elided subject.  The
    antecedent is in a neighbouring sentence or the section heading.

other
    None of the above (unused -- every gold link fell into one of the four).

Entry format
------------
    (project, sentence, component): (form, evidence)

`evidence` quotes the exact span that carries the reference, or says why there
is none.  Where the verdict is contestable the note says so explicitly; nothing
is hidden behind the label.
"""

FORMS = {
    "canonical": "the catalog name itself (modulo case/space/hyphen/camelCase/plural)",
    "alias": "a different document string (expansion, contraction, package or substitute name)",
    "partial": "only part of a multi-word name (distinctive word or generic head noun)",
    "pronoun_or_implicit": "no form of the name; pronoun, demonstrative or elided subject",
    "other": "none of the above",
}

#: which proposal form of our approach SHOULD carry a link of each true form
EXPECTED_PROPOSAL = {
    "canonical": {"full_name"},
    "alias": {"full_name"},            # the alias table feeds the full-name scan
    "partial": {"partial_name"},
    "pronoun_or_implicit": {"coreference"},
    "other": set(),
}

ANNOTATIONS = {
 # ── mediastore ──────────────────────────────────────────────────────────────
 ("mediastore",1,"Facade"): ("canonical","'namely the Facade component'"),
 ("mediastore",3,"Facade"): ("canonical","'the Facade component delivers'"),
 ("mediastore",6,"Facade"): ("canonical","'using the Facade component'"),
 ("mediastore",7,"MediaManagement"): ("canonical","'called the MediaManagement component'"),
 ("mediastore",8,"MediaManagement"): ("canonical","'The MediaManagement component coordinates'"),
 ("mediastore",9,"MediaManagement"): ("pronoun_or_implicit","'Furthermore, it fetches audio files' - 'it' = the MediaManagement component of s8"),
 ("mediastore",11,"UserManagement"): ("canonical","'The UserManagement component answers'"),
 ("mediastore",12,"UserDBAdapter"): ("canonical","'The UserDBAdapter component queries'"),
 ("mediastore",13,"UserManagement"): ("canonical","'the UserManagement component implements'"),
 ("mediastore",16,"TagWatermarking"): ("canonical","'watermarked by the TagWatermarking component'"),
 ("mediastore",17,"TagWatermarking"): ("canonical","'from the TagWatermarking component'"),
 ("mediastore",17,"MediaManagement"): ("canonical","'the MediaManagement component forwards'"),
 ("mediastore",19,"Packaging"): ("canonical","'we provide the Packaging component'"),
 ("mediastore",20,"Reencoding"): ("alias","'The ReEncoder component converts' - catalog name is Reencoding; the doc never writes that form, it uses the agent noun"),
 ("mediastore",23,"DB"): ("alias","'The Database component represents an actual database' - expansion of the catalog abbreviation DB"),
 ("mediastore",24,"DB"): ("pronoun_or_implicit","'It stores user information' - 'It' = the Database component of s23"),
 ("mediastore",25,"DB"): ("alias","'a query that is sent to the Database component'"),
 ("mediastore",25,"MediaAccess"): ("alias","'AudioAccess creates a query' - document-only name; no AudioAccess exists in the catalog, the component is MediaAccess"),
 ("mediastore",26,"MediaAccess"): ("canonical","'the MediaAccess component stores it'"),
 ("mediastore",27,"MediaAccess"): ("canonical","'The MediaAccess component encapsulates'"),
 ("mediastore",28,"MediaAccess"): ("pronoun_or_implicit","'Furthermore, it fetches a list' - 'it' = the MediaAccess component of s27"),
 ("mediastore",29,"UserDBAdapter"): ("canonical","'the UserDBAdapter component provides'"),
 ("mediastore",30,"UserDBAdapter"): ("canonical","'The UserDBAdapter component creates'"),
 ("mediastore",31,"DB"): ("alias","'The Database component then executes'"),
 ("mediastore",32,"DB"): ("alias","'also stored in the Database component'"),
 ("mediastore",33,"DB"): ("alias","'decouple the DataStorage from the database' - lower-case use of the same expansion"),
 ("mediastore",33,"FileStorage"): ("alias","'decouple the DataStorage' - document-only name for the FileStorage component"),
 ("mediastore",34,"DB"): ("alias","'fetches the associated meta-data from the Database'"),
 ("mediastore",34,"MediaAccess"): ("canonical","'the MediaAccess component fetches'"),
 ("mediastore",35,"FileStorage"): ("alias","'retrieved from the DataStorage'"),
 ("mediastore",36,"FileStorage"): ("alias","'stored in the DataStorage without any change'"),

 # ── teastore ────────────────────────────────────────────────────────────────
 ("teastore",1,"Registry"): ("canonical","'a single Registry instance'"),
 ("teastore",2,"WebUI"): ("canonical","'The WebUI service retrieves'"),
 ("teastore",2,"ImageProvider"): ("canonical","'from the Image Provider' - the catalog name ImageProvider written with a space; normalisation alone resolves it"),
 ("teastore",3,"Auth"): ("canonical","'authenticated by the Auth service'"),
 ("teastore",4,"Persistence"): ("canonical","'retrieved from the PersistenceProvider' - compound CONTAINING the catalog name Persistence"),
 ("teastore",4,"Recommender"): ("canonical","'from the Recommender service'"),
 ("teastore",5,"WebUI"): ("canonical","'The WebUI provides the TeaStore front-end'"),
 ("teastore",6,"WebUI"): ("pronoun_or_implicit","'It contains logic to save and retireve values from cookies' - 'It' = the WebUI of s5"),
 ("teastore",7,"WebUI"): ("canonical","'not provides by the WebUi' - case variant"),
 ("teastore",7,"ImageProvider"): ("canonical","'from the Image Provider service'"),
 ("teastore",8,"WebUI"): ("partial","'The UI provides a status page' - only the second half of WebUI; note that teammates has a component actually NAMED UI"),
 ("teastore",10,"WebUI"): ("canonical","'delivers images to the WebUI'"),
 ("teastore",10,"ImageProvider"): ("canonical","'The Image Provider delivers images'"),
 ("teastore",11,"ImageProvider"): ("pronoun_or_implicit","'It matches the provided product ID' - 'It' = the Image Provider of s10"),
 ("teastore",12,"ImageProvider"): ("canonical","'not available to the Image Provider'"),
 ("teastore",18,"Auth"): ("canonical","'The Auth service handles'"),
 ("teastore",22,"Persistence"): ("canonical","'The Persistence service provides'"),
 ("teastore",23,"Persistence"): ("pronoun_or_implicit","'It maps the relational entities' - 'It' = the Persistence service of s22"),
 ("teastore",24,"Persistence"): ("pronoun_or_implicit","'It features endpoints for general CRUD-Operations'"),
 ("teastore",25,"Persistence"): ("canonical","'The persistence provider uses a second level entity cache' - lower-case spaced variant"),
 ("teastore",26,"Persistence"): ("pronoun_or_implicit","'As such, it also acts as a caching layer'"),
 ("teastore",27,"Recommender"): ("canonical","'The Recommender is used to generate'"),
 ("teastore",28,"Recommender"): ("pronoun_or_implicit","'It is trained using all existing orders'"),
 ("teastore",37,"Registry"): ("canonical","'The Registry provides information'"),
 ("teastore",38,"Registry"): ("canonical","'register themselves at the registry on startup'"),
 ("teastore",41,"Registry"): ("canonical","'uses one single registry'"),
 ("teastore",43,"Registry"): ("canonical","'By limiting it to a single registry instance'"),

 # ── teammates ───────────────────────────────────────────────────────────────
 ("teammates",1,"UI"): ("canonical","'Architecture contains UI Component'"),
 ("teammates",1,"Logic"): ("canonical","'Logic Component'"),
 ("teammates",1,"Storage"): ("canonical","'Storage Component'"),
 ("teammates",1,"Test Driver"): ("canonical","'Test Driver Component'"),
 ("teammates",1,"E2E"): ("canonical","'E2E Component'"),
 ("teammates",1,"Client"): ("canonical","'Client Component'"),
 ("teammates",1,"Common"): ("canonical","'Common Component'"),
 ("teammates",4,"UI"): ("canonical","'The UI Browser seen by users' - compound containing the name"),
 ("teammates",5,"UI"): ("canonical","'This UI is a single HTML page'"),
 ("teammates",7,"UI"): ("canonical","'In the UI Server the entry point'"),
 ("teammates",7,"Logic"): ("canonical","'the application back end logic' - the name word, used as a common noun"),
 ("teammates",8,"Logic"): ("canonical","'The main logic of the application is in POJOs' - common-noun use of the name"),
 ("teammates",9,"Storage"): ("canonical","'The storage layer of the application' - common-noun use of the name"),
 ("teammates",9,"GAE Datastore"): ("canonical","'provided by GAE Datastore, a NoSQL database'"),
 ("teammates",10,"Test Driver"): ("canonical","'The following explains the use of the Test Driver'"),
 ("teammates",15,"E2E"): ("canonical","'The E2E end-to-end component'"),
 ("teammates",16,"E2E"): ("canonical","'Its primary function is for E2E tests' - the name occurs; the component-denoting subject is nevertheless the anaphor 'Its'"),
 ("teammates",18,"Client"): ("canonical","'The Client component can connect'"),
 ("teammates",19,"Client"): ("pronoun_or_implicit","'It is used for administrative purposes' - 'It' = the Client component of s18"),
 ("teammates",20,"Common"): ("canonical","'The Common component contains utility code'"),
 ("teammates",25,"UI"): ("canonical","'the object structure of the UI component'"),
 ("teammates",29,"UI"): ("canonical","'The UI component is the first stop'"),
 ("teammates",47,"Logic"): ("canonical","'interacting with the Logic component as necessary'"),
 ("teammates",68,"Logic"): ("canonical","'interacting with the Logic component as necessary'"),
 ("teammates",77,"Logic"): ("canonical","'The Logic component handles the business logic'"),
 ("teammates",78,"Logic"): ("pronoun_or_implicit","'In particular, it is responsible for the following' - 'it' = the Logic component of s77"),
 ("teammates",81,"UI"): ("canonical","'received from the UI component'"),
 ("teammates",85,"UI"): ("canonical","'to be accessed by the UI'"),
 ("teammates",87,"Logic"): ("canonical","'Logic API is represented by the classes Logic, GateKeeper' - the name also happens to be a class name here"),
 ("teammates",88,"Logic"): ("canonical","'connects to the several Logic classes' - 'Logic' here names a Facade CLASS inside the component"),
 ("teammates",88,"Storage"): ("canonical","'to access data from the Storage component'"),
 ("teammates",97,"UI"): ("canonical","'The UI is expected to check access control'"),
 ("teammates",97,"Logic"): ("canonical","'before calling a method in the Logic'"),
 ("teammates",101,"Storage"): ("canonical","'(escalated from Storage level)'"),
 ("teammates",118,"Storage"): ("canonical","'The Storage component performs CRUD'"),
 ("teammates",119,"Storage"): ("pronoun_or_implicit","'It contains minimal logic beyond what is directly relevant to CRUD' - 'It' = the Storage component of s118"),
 ("teammates",120,"Storage"): ("pronoun_or_implicit","'In particular, it is reponsible for the following'"),
 ("teammates",122,"Logic"): ("canonical","'from the Logic component'"),
 ("teammates",122,"GAE Datastore"): ("partial","'Hiding the complexities of datastore' - only the head word of the two-word name GAE Datastore"),
 ("teammates",123,"Storage"): ("canonical","'contained inside the Storage component'"),
 ("teammates",128,"Storage"): ("canonical","'The Storage component does not perform'"),
 ("teammates",129,"Logic"): ("canonical","'handled by the Logic component'"),
 ("teammates",131,"Logic"): ("canonical","'to be accessed by the logic component'"),
 ("teammates",137,"GAE Datastore"): ("canonical","'act as the bridge to the GAE Datastore'"),
 ("teammates",138,"GAE Datastore"): ("partial","'until data is persisted in the datastore' - head word only"),
 ("teammates",141,"GAE Datastore"): ("partial","\"across all serves of the Google's distributed datastore\" - head word plus a descriptive paraphrase of GAE"),
 ("teammates",155,"Common"): ("canonical","'The Common component contains common utilities'"),
 ("teammates",163,"Test Driver"): ("canonical","'Test Driver can use the DataBundle'"),
 ("teammates",168,"Test Driver"): ("pronoun_or_implicit","'This component automates the testing of TEAMMATES' - demonstrative shell; the component is identified only by the section (s169 'test.driver, test.cases')"),
 ("teammates",174,"Common"): ("canonical","'the datatransfer objects from the Common component'"),
 ("teammates",175,"Common"): ("canonical","'the utility classes from the Common component'"),
 ("teammates",176,"Logic"): ("canonical","'for testing the Logic component'"),
 ("teammates",177,"Storage"): ("canonical","'for testing the Storage component'"),
 ("teammates",185,"Logic"): ("canonical","'REST API calls for the back-end logic' - common-noun use of the name"),
 ("teammates",185,"E2E"): ("canonical","'The E2E component has no knowledge'"),
 ("teammates",186,"E2E"): ("canonical","'Its primary function is for E2E tests and L&P tests' - name present, subject is the anaphor 'Its'"),
 ("teammates",194,"Client"): ("canonical","'The Client component contains scripts'"),

 # ── bigbluebutton ───────────────────────────────────────────────────────────
 ("bigbluebutton",4,"HTML5 Client"): ("canonical","'HTML5 client.' - section heading"),
 ("bigbluebutton",5,"HTML5 Client"): ("canonical","'The HTML5 client is a single page, responsive web application'"),
 ("bigbluebutton",6,"HTML5 Client"): ("canonical","'The HTML5 client connects directly'"),
 ("bigbluebutton",6,"HTML5 Server"): ("partial","'with the BigBlueButton server over port 443' - generic head noun only; the 'html5' the auto classifier matched belongs to the CLIENT mention in the same sentence"),
 ("bigbluebutton",8,"HTML5 Server"): ("canonical","'The HTML5 server sits behind nginx'"),
 ("bigbluebutton",9,"HTML5 Client"): ("partial","'the state of each BigBlueButton client' - generic head noun; the sentence's 'HTML5' belongs to the SERVER mention"),
 ("bigbluebutton",9,"HTML5 Server"): ("canonical","'The HTML5 server is built upon Meteor.js'"),
 ("bigbluebutton",10,"HTML5 Client"): ("partial","'each client connected to a meeting' - generic head noun only"),
 ("bigbluebutton",10,"HTML5 Server"): ("partial","'all meetings on the server' - generic head noun only"),
 ("bigbluebutton",11,"HTML5 Client"): ("partial","\"Each user's client is only aware of the their meeting's state\""),
 ("bigbluebutton",12,"HTML5 Client"): ("partial","'The client side subscribes'"),
 ("bigbluebutton",12,"HTML5 Server"): ("partial","'the published collections on the server side'"),
 ("bigbluebutton",13,"HTML5 Client"): ("partial","'pushed to MiniMongo on the client side'"),
 ("bigbluebutton",13,"HTML5 Server"): ("partial","'Updates to MongoDB on the server side'"),
 ("bigbluebutton",14,"HTML5 Client"): ("canonical","'the architecture of the HTML5 client'"),
 ("bigbluebutton",15,"HTML5 Server"): ("canonical","'Scalability of HTML5 server component.' - section heading"),
 ("bigbluebutton",19,"HTML5 Client"): ("partial","'handling incoming messages from clients' - generic head noun, plural; the 'html5' in this sentence is inside 'bbb-html5', which names the SERVER"),
 ("bigbluebutton",19,"HTML5 Server"): ("alias","'a single nodejs process for bbb-html5' - the deployment/package name, never the catalog form"),
 ("bigbluebutton",20,"HTML5 Server"): ("alias","'bbb-html5 could use multiple CPU cores'"),
 ("bigbluebutton",21,"HTML5 Server"): ("alias","'bbb-html5 uses 2 \"frontend\" and two \"backend\" processes'"),
 ("bigbluebutton",26,"Apps"): ("canonical","'send events to akka-apps' - compound containing the catalog name Apps"),
 ("bigbluebutton",30,"BBB web"): ("canonical","'bbb-web splits the load in round-robin fashion' - hyphenated form of the two-word name BBB web"),
 ("bigbluebutton",36,"BBB web"): ("canonical","'BBB web.' - section heading"),
 ("bigbluebutton",37,"BBB web"): ("alias","'BigBlueButton web application is a Java-based application' - expansion of the BBB abbreviation"),
 ("bigbluebutton",38,"BBB web"): ("pronoun_or_implicit","'It implements the BigBlueButton API' - 'It' = the BigBlueButton web application of s37"),
 ("bigbluebutton",39,"HTML5 Server"): ("partial","'an endpoint to control the BigBlueButton server' - generic head noun; arguably the whole deployment rather than the HTML5 server component, so the gold link itself is debatable"),
 ("bigbluebutton",46,"Redis PubSub"): ("canonical","'Redis PubSub.' - section heading"),
 ("bigbluebutton",47,"Redis PubSub"): ("canonical","'Redis PubSub provides a communication channel'"),
 ("bigbluebutton",47,"HTML5 Server"): ("partial","'running on the BigBlueButton server' - generic head noun; same debatable gold reading as s39"),
 ("bigbluebutton",48,"Redis DB"): ("canonical","'Redis DB.' - section heading"),
 ("bigbluebutton",49,"Redis DB"): ("canonical","'all events are stored in Redis DB'"),
 ("bigbluebutton",51,"Apps"): ("canonical","'Apps akka.' - section heading"),
 ("bigbluebutton",52,"Apps"): ("canonical","'BigBlueButton Apps is the main application'"),
 ("bigbluebutton",53,"Apps"): ("pronoun_or_implicit","'It provides the list of users, chat, whiteboard' - 'It' = BigBlueButton Apps of s52"),
 ("bigbluebutton",54,"Apps"): ("canonical","'the different components of Apps Akka'"),
 ("bigbluebutton",57,"FSESL"): ("canonical","'FSESL akka.' - section heading"),
 ("bigbluebutton",58,"FreeSWITCH"): ("canonical","'the component that integrates with FreeSWITCH'"),
 ("bigbluebutton",59,"FreeSWITCH"): ("canonical","'voice conference systems other than FreeSWITCH'"),
 ("bigbluebutton",60,"Redis PubSub"): ("canonical","'uses messages through redis pubsub'"),
 ("bigbluebutton",60,"FSESL"): ("alias","'FreeSWITCH Event Socket Layer (fsels)' - the expansion; the parenthetical abbreviation is even mis-spelled ('fsels' vs FSESL), so the catalog form never appears"),
 ("bigbluebutton",60,"Apps"): ("canonical","'Communication between apps and FreeSWITCH'"),
 ("bigbluebutton",61,"FreeSWITCH"): ("canonical","'FreeSWITCH.' - section heading"),
 ("bigbluebutton",62,"FreeSWITCH"): ("canonical","'We think FreeSWITCH is an amazing piece of software'"),
 ("bigbluebutton",63,"FreeSWITCH"): ("canonical","'FreeSWITCH provides the voice conferencing capability'"),
 ("bigbluebutton",65,"WebRTC-SFU"): ("partial","'by connecting using WebRTC' - the distinctive half of WebRTC-SFU, but here it denotes the PROTOCOL; the gold link to the component is debatable"),
 ("bigbluebutton",66,"FreeSWITCH"): ("canonical","'FreeSWITCH can also be integrated with VOIP providers'"),
 ("bigbluebutton",67,"kurento"): ("canonical","'Kurento and WebRTC-SFU.' - section heading"),
 ("bigbluebutton",67,"WebRTC-SFU"): ("canonical","'Kurento and WebRTC-SFU.' - section heading"),
 ("bigbluebutton",68,"kurento"): ("canonical","'Kurento Media Server KMS is a media server'"),
 ("bigbluebutton",69,"kurento"): ("alias","'KMS is responsible for streaming of webcams' - the abbreviation introduced in s68"),
 ("bigbluebutton",70,"WebRTC-SFU"): ("canonical","'The WebRTC-SFU acts as the media controller'"),
 ("bigbluebutton",72,"HTML5 Client"): ("canonical","'from the BigBlueButton HTML5 client'"),
 ("bigbluebutton",72,"FreeSWITCH"): ("canonical","'the voice conference (running in FreeSWITCH)'"),
 ("bigbluebutton",73,"HTML5 Client"): ("partial","'the BigBlueButton client will make an audio connection' - generic head noun"),
 ("bigbluebutton",73,"WebRTC-SFU"): ("partial","'an audio connection to the server via WebRTC' - protocol reading again; debatable gold link"),
 ("bigbluebutton",73,"HTML5 Server"): ("partial","'an audio connection to the server'"),
 ("bigbluebutton",76,"HTML5 Client"): ("partial","'to be displayed inside the client'"),
 ("bigbluebutton",78,"BBB web"): ("canonical","'converted into scalable vector graphics (SVG) via bbb-web'"),
 ("bigbluebutton",79,"HTML5 Client"): ("partial","'sends progress messages to the client'"),
 ("bigbluebutton",79,"Redis PubSub"): ("canonical","'through the Redis pubsub'"),
 ("bigbluebutton",80,"Presentation Conversion"): ("canonical","'Presentation conversion flow.' - section heading"),
 ("bigbluebutton",81,"Presentation Conversion"): ("canonical","'the flow of the presentation conversion'"),

 # ── jabref ──────────────────────────────────────────────────────────────────
 ("jabref",1,"gui"): ("canonical","'towards the gui which is the outer shell'"),
 ("jabref",1,"logic"): ("canonical","'the logic as an intermediate layer'"),
 ("jabref",1,"model"): ("canonical","'with the model in the center'"),
 ("jabref",2,"cli"): ("canonical","'utility packages for preferences and the cli'"),
 ("jabref",2,"preferences"): ("canonical","'utility packages for preferences'"),
 ("jabref",4,"gui"): ("canonical","'(between logic, model, and gui)'"),
 ("jabref",4,"logic"): ("canonical","'(between logic, model, and gui)'"),
 ("jabref",4,"model"): ("canonical","'(between logic, model, and gui)'"),
 ("jabref",5,"model"): ("canonical","'The model represents the most important data structures'"),
 ("jabref",6,"gui"): ("canonical","'an API the gui can call and use'"),
 ("jabref",6,"logic"): ("canonical","'The logic is responsible for reading/writing'"),
 ("jabref",6,"model"): ("canonical","'manipulating the model'"),
 ("jabref",7,"gui"): ("canonical","'Only the gui knows the user'"),
 ("jabref",9,"logic"): ("canonical","'the logic should only depend on model classes'"),
 ("jabref",9,"model"): ("canonical","'The model should have no dependencies'"),
 ("jabref",10,"cli"): ("canonical","'The cli package bundles classes'"),
 ("jabref",11,"preferences"): ("canonical","'The preferences represents all information customizable by a user'"),
 ("jabref",12,"model"): ("canonical","'publish events from the model to the other layers'"),
}
