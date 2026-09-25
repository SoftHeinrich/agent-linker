#!/usr/bin/env python3
"""MANUAL adjudication of the CH2 'type 2' false positives, one by one.

Type 2 = a false positive on a sentence carrying NO surface form of the wrongly
linked component. `audit.py --sheets DIR` derives the population
(`ch2_mode2_sheet.csv`, 106 items); this file records a human verdict for each of
them and nothing else. Every verdict was reached by reading the sentence and its
neighbours in the SAD.

Sub-modes (one per item, the dominant one):
  a  topic continuation   the component is named in an adjacent sentence and this
                          one elaborates without naming it
  b  long-range drift     the nearest sentence gold links to that component is far
                          away; the link is inferred from general topic
  d  sibling confusion    linked to a sibling, parent or child of the right component
  e  role-word confusion  a document word ("frontend", "backend") denotes a process
                          role here, not the catalog component of that name
  f  unlinked component   the component carries no gold link anywhere in the project
  g  CLASSIFIER ARTIFACT  a surface form IS present (plural, hyphenated or camelCase
                          variant) and the automatic tokenizer missed it -- so the
                          item is really a type-1 (lexical) false positive

gold_silent: does the sentence actually describe that component's responsibility,
so that the gold standard, not the system, is arguably what is wrong?
  yes | borderline | no
"""

# (project, sentence, wrong_component): (sub-mode, gold_silent, note)
ANNOTATIONS = {
 ("bigbluebutton",7,"HTML5 Client"): ("a","no","sentence is about nginx; the client is named in s6"),
 ("bigbluebutton",16,"HTML5 Server"): ("a","borderline","s15 is the heading 'Scalability of HTML5 server component'; s16 describes its 2.2 history"),
 ("bigbluebutton",17,"HTML5 Server"): ("a","borderline","same section; describes the nodejs bottleneck of that server"),
 ("bigbluebutton",23,"HTML5 Client"): ("e","no","'front-end/back-end' here are nodejs process roles inside the HTML5 server"),
 ("bigbluebutton",24,"HTML5 Client"): ("e","no","'Frontends' = nodejs frontend processes, not the HTML5 Client component"),
 ("bigbluebutton",25,"HTML5 Client"): ("e","no","'Frontends collect subscriptions'"),
 ("bigbluebutton",26,"HTML5 Client"): ("e","no","'Frontends receive other DDP events'; gold here is Apps"),
 ("bigbluebutton",27,"HTML5 Client"): ("e","no","'Frontends handle the Streamer redis events'"),
 ("bigbluebutton",28,"HTML5 Client"): ("e","no","'Frontends still require MeetingStarted'"),
 ("bigbluebutton",29,"HTML5 Client"): ("e","no","'Backends handle all the non-streamer events'"),
 ("bigbluebutton",30,"kurento"): ("b","no","sentence is bbb-web load splitting; kurento is 37 sentences away"),
 ("bigbluebutton",31,"Apps"): ("e","no","'individual backends' = process roles"),
 ("bigbluebutton",31,"BBB web"): ("a","no","bbb-web named in s30; this sentence is about backends"),
 ("bigbluebutton",32,"Apps"): ("e","no","'passed to backends as well'"),
 ("bigbluebutton",32,"HTML5 Client"): ("e","no","'no frontends'"),
 ("bigbluebutton",41,"BBB web"): ("b","no","third-party integrations, not the web application"),
 ("bigbluebutton",42,"BBB web"): ("e","no","'its own front-end called Greenlight' is a third-party portal"),
 ("bigbluebutton",42,"WebRTC-SFU"): ("b","no","unrelated; 23 sentences from the nearest SFU sentence"),
 ("bigbluebutton",43,"Recording Service"): ("f","no","Recording Service carries no gold link anywhere; the NAME scan proposed it with no surface form"),
 ("bigbluebutton",44,"BBB web"): ("b","no","'simple API demos'"),
 ("bigbluebutton",45,"BBB web"): ("b","borderline","'they all use the API under the hood' -- the API is bbb-web's"),
 ("bigbluebutton",49,"Recording Service"): ("f","borderline","'when a meeting is recorded'; the component is never in gold"),
 ("bigbluebutton",53,"Presentation Conversion"): ("g","no","'presentations' is the plural of the name's head word; the tokenizer required the exact form"),
 ("bigbluebutton",56,"Apps"): ("a","no","MeetingActor named in s55"),
 ("bigbluebutton",57,"FreeSWITCH"): ("d","no","sibling of FSESL, which is what s57 heads"),
 ("bigbluebutton",58,"FSESL"): ("d","yes","'the component that integrates with FreeSWITCH' IS FSESL; gold assigns FreeSWITCH"),
 ("bigbluebutton",59,"FSESL"): ("d","borderline","about replacing FreeSWITCH; gold assigns FreeSWITCH"),
 ("bigbluebutton",78,"Presentation Conversion"): ("d","borderline","PDF->SVG conversion is presentation conversion; gold assigns bbb-web"),
 ("jabref",3,"model"): ("a","borderline","layer dependency direction; OUR ONLY SURVIVING TYPE-2 FP"),
 ("jabref",8,"gui"): ("a","no","'for each layer we form packages'; gui named in s7"),
 ("jabref",8,"logic"): ("a","no","same sentence, same reason"),
 ("jabref",8,"model"): ("a","no","same sentence, same reason"),
 ("mediastore",8,"UserManagement"): ("b","no","sentence is about MediaManagement coordinating others"),
 ("mediastore",15,"MediaManagement"): ("a","no","'the requested files are first reencoded'"),
 ("mediastore",16,"AudioWatermarking"): ("d","no","sibling of TagWatermarking; never linked in gold"),
 ("mediastore",16,"ParallelWatermarking"): ("d","no","sibling of TagWatermarking; never linked in gold"),
 ("mediastore",17,"AudioWatermarking"): ("d","no","sibling of TagWatermarking"),
 ("mediastore",17,"ParallelWatermarking"): ("d","no","sibling of TagWatermarking"),
 ("mediastore",21,"Reencoding"): ("a","borderline","'this can result in reduction of file sizes' continues the ReEncoder description"),
 ("mediastore",26,"FileStorage"): ("d","borderline","'stores it at the predefined location'; gold assigns MediaAccess"),
 ("mediastore",37,"Reencoding"): ("g","no","'re-encoding' is the hyphenated form of the catalog name -- SWATTR's only type-2 item"),
 ("teammates",3,"UI"): ("a","no","'overview of the main components'"),
 ("teammates",11,"Test Driver"): ("a","borderline","s10 heads 'the use of the Test Driver'; s11 is that section"),
 ("teammates",13,"Test Driver"): ("a","borderline","testing-framework list under the Test Driver section"),
 ("teammates",14,"Test Driver"): ("a","borderline","same section"),
 ("teammates",16,"Test Driver"): ("d","no","gold assigns E2E, a sibling test component"),
 ("teammates",17,"Test Driver"): ("d","borderline","Selenium E2E automation; gold assigns E2E"),
 ("teammates",21,"Common"): ("a","no","'the diagram below shows how code is organized'"),
 ("teammates",27,"UI"): ("a","borderline","'written in Angular' continues the ui.website description"),
 ("teammates",30,"UI"): ("a","no","'such a request will go through the following steps'"),
 ("teammates",36,"Client"): ("b","no","Client is the admin CLI; this is browser traffic, 17 sentences away"),
 ("teammates",36,"UI"): ("b","borderline","user requests from the web browser"),
 ("teammates",38,"UI"): ("b","no","'the initial request for the web page'"),
 ("teammates",40,"UI"): ("b","borderline","'WebPageServlet returns the built single web page'"),
 ("teammates",41,"Client"): ("b","no","browser rendering, not the admin Client"),
 ("teammates",42,"GAE Datastore"): ("b","no","AJAX request processing, 33 sentences from the nearest datastore sentence"),
 ("teammates",45,"Logic"): ("a","no","'WebApiServlet executes the action'"),
 ("teammates",47,"UI"): ("b","no","gold assigns Logic; the sentence names Logic"),
 ("teammates",48,"Logic"): ("a","no","'the Action packages the result'"),
 ("teammates",50,"Client"): ("b","no","'sends the result back to the browser'"),
 ("teammates",52,"UI"): ("b","no","'the Web API is protected by two layers'"),
 ("teammates",53,"UI"): ("b","no","'origin check, authentication and authorization'"),
 ("teammates",55,"UI"): ("b","no","access-control description"),
 ("teammates",56,"Test Driver"): ("b","no","'typically for testing purpose' 46 sentences away"),
 ("teammates",59,"GAE Datastore"): ("b","no","'this type of request will be processed as follows'"),
 ("teammates",64,"Test Driver"): ("b","no","'useful in testing the actions' 54 sentences away"),
 ("teammates",67,"Logic"): ("a","no","'automatedServlet executes the action'"),
 ("teammates",72,"GAE Datastore"): ("b","no","'configured in cron.xml'"),
 ("teammates",75,"GAE Datastore"): ("b","no","'configured in queue.xml'"),
 ("teammates",92,"Logic"): ("a","borderline","EmailSender is a Logic class; gold silent on the elaboration"),
 ("teammates",94,"Logic"): ("a","borderline","TaskQueuer is a Logic class"),
 ("teammates",96,"Logic"): ("a","yes","'this component provides methods to perform access control' refers to Logic"),
 ("teammates",121,"Storage"): ("a","yes","a responsibility bullet under the Storage section"),
 ("teammates",122,"Storage"): ("a","yes","'hiding the complexities of datastore from the Logic component' is Storage's own responsibility"),
 ("teammates",137,"Storage"): ("d","borderline","the Db classes are Storage; gold assigns GAE Datastore"),
 ("teammates",182,"Test Driver"): ("b","borderline","'front-end files are tested with Jest'"),
 ("teammates",184,"Test Driver"): ("b","borderline","'how TEAMMATES testing maps to standard types'"),
 ("teammates",186,"Test Driver"): ("d","no","gold assigns E2E"),
 ("teammates",188,"Test Driver"): ("d","borderline","'e2e.util helpers for running E2E tests'"),
 ("teammates",193,"Test Driver"): ("b","borderline","'load and performance tests'"),
 ("teastore",4,"WebUI"): ("a","borderline","the implicit retriever is the WebUI; gold assigns Persistence and Recommender"),
 ("teastore",9,"WebUI"): ("a","yes","'the status view' is the status page the UI provides in s8"),
 ("teastore",13,"ImageProvider"): ("a","yes","image lookup fallback -- an Image Provider responsibility"),
 ("teastore",14,"ImageProvider"): ("a","yes","image scaling -- an Image Provider responsibility"),
 ("teastore",15,"ImageProvider"): ("a","yes","'the scaled image is stored for later use'"),
 ("teastore",16,"ImageProvider"): ("a","yes","the LFU image cache"),
 ("teastore",17,"ImageProvider"): ("a","yes","cache lookup before loading from disk"),
 ("teastore",19,"Auth"): ("a","yes","'passwords are hashed using BCrypt' -- an Auth responsibility"),
 ("teastore",20,"Auth"): ("a","yes","session validation via SessionBlob"),
 ("teastore",21,"Auth"): ("a","yes","session-tampering check"),
 ("teastore",29,"Recommender"): ("a","yes","how recommendations are generated"),
 ("teastore",30,"Recommender"): ("a","yes","item rating basis"),
 ("teastore",31,"Recommender"): ("a","yes","fallback algorithm for unknown users"),
 ("teastore",32,"Recommender"): ("a","yes","Slope One collaborative filtering"),
 ("teastore",32,"SlopeOneRecommender"): ("g","no","'Slope One' IS written; the tokenizer does not split the camelCase catalog name"),
 ("teastore",33,"Recommender"): ("a","yes","'we implemented two versions of the algorithm'"),
 ("teastore",33,"SlopeOneRecommender"): ("g","no","camelCase catalog name vs 'Slope One' in the text"),
 ("teastore",34,"PreprocessedSlopeOneRecommender"): ("g","no","the two versions described ARE the preprocessed/on-the-go variants"),
 ("teastore",34,"Recommender"): ("a","yes","CPU- vs memory-intensive variants"),
 ("teastore",34,"SlopeOneRecommender"): ("g","no","camelCase catalog name"),
 ("teastore",35,"OrderBasedRecommender"): ("g","no","'order-based nearest-neighbor approach' IS the name, hyphenated and split"),
 ("teastore",35,"Recommender"): ("a","yes","the order-based variant"),
 ("teastore",36,"OrderBasedRecommender"): ("g","no","'its recommendation time' continues the order-based description"),
 ("teastore",36,"Recommender"): ("a","yes","recommendation timing"),
 ("teastore",39,"Registry"): ("a","yes","'services send a heartbeat by re-registering' -- a Registry responsibility"),
 ("teastore",40,"Registry"): ("a","yes","heartbeat timeout handling"),
}

SUBMODE = {
 "a": "topic continuation (named in an adjacent sentence)",
 "b": "long-range topical drift",
 "d": "sibling / parent / child confusion",
 "e": "role-word confusion (a document word, not the component)",
 "f": "component carries no gold link anywhere",
 "g": "CLASSIFIER ARTIFACT -- a surface form is present; really a type-1 FP",
}
