// Build an editable CARE diagram with native Excalidraw elements.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const output = resolve(dirname(fileURLToPath(import.meta.url)),
  "..", "paper_asem", "figures", "care_architecture.excalidraw");
const C = {
  paper:"#fffefa", ink:"#27374a", muted:"#64748b",
  teal:"#1d8b82", blue:"#386fb0", violet:"#735da8",
  amber:"#b87937", coral:"#c25e63", green:"#318663",
  paleTeal:"#edf9f5", paleBlue:"#f0f6ff",
  paleViolet:"#f6f2fc", paleAmber:"#fff7ea",
  paleCoral:"#fff2f0", paleGreen:"#eaf7ef",
};
let serial=0;
const elements=[];
function common(type,x,y,width,height,extra={}) {
  serial++;
  return {
    id:"carex"+String(serial).padStart(4,"0"),
    type,x,y,width,height,angle:0,
    strokeColor:C.ink,backgroundColor:"transparent",
    fillStyle:"solid",strokeWidth:2,strokeStyle:"solid",
    roughness:1,opacity:100,groupIds:[],frameId:null,
    roundness:null,seed:41003+serial*193,
    version:1,versionNonce:83011+serial*227,
    isDeleted:false,boundElements:null,updated:1794500000000,
    link:null,locked:false,...extra,
  };
}
function add(e) { elements.push(e); return e; }
function shape(type,x,y,w,h,color,fill="transparent",extra={}) {
  return add(common(type,x,y,w,h,{
    strokeColor:color,backgroundColor:fill,
    roundness:type==="rectangle"?{type:3}:null,...extra,
  }));
}
function text(value,x,y,w,h,size,color=C.ink,align="center") {
  return add(common("text",x,y,w,h,{
    strokeColor:color,strokeWidth:1,roughness:0,
    text:value,originalText:value,
    fontSize:size,fontFamily:5,textAlign:align,
    verticalAlign:"middle",lineHeight:1.25,
    autoResize:false,containerId:null,
  }));
}
function path(type,points,color,width=3) {
  const [x,y]=points[0], rel=points.map(p=>[p[0]-x,p[1]-y]);
  return add(common(type,x,y,
    Math.max(...rel.map(p=>p[0]))-Math.min(...rel.map(p=>p[0])),
    Math.max(...rel.map(p=>p[1]))-Math.min(...rel.map(p=>p[1])),{
      strokeColor:color,strokeWidth:width,roughness:0.75,
      roundness:{type:2},points:rel,lastCommittedPoint:null,
      startArrowhead:null,endArrowhead:type==="arrow"?"arrow":null,
    }));
}
function line(a,b,color,width=2) { return path("line",[a,b],color,width); }
function pill(value,x,y,w,h,color,fill=C.paper,size=18) {
  shape("rectangle",x,y,w,h,color,fill,{strokeWidth:1.7});
  text(value,x+7,y+3,w-14,h-6,size,color);
}
function circle(cx,cy,color,fill) {
  shape("ellipse",cx-51,cy-51,102,102,color,fill,
    {strokeWidth:3.1,roughness:1.1});
}
function source(cx,cy) {
  circle(cx,cy,C.teal,C.paleTeal);
  shape("rectangle",cx-23,cy-27,47,54,C.teal,C.paper,{strokeWidth:2.2});
  for(const dy of [-14,-2,10]) line([cx-13,cy+dy],[cx+12,cy+dy],C.teal,2);
}
function rank(cx,cy) {
  circle(cx,cy,C.blue,C.paleBlue);
  shape("ellipse",cx-25,cy-25,40,40,C.blue,C.paper,{strokeWidth:2.4});
  line([cx+8,cy+8],[cx+29,cy+29],C.blue,3);
}
function bundle(cx,cy) {
  circle(cx,cy,C.blue,C.paleBlue);
  shape("rectangle",cx-14,cy-28,43,45,C.blue,C.paper,{strokeWidth:2});
  shape("rectangle",cx-28,cy-14,43,45,C.blue,C.paper,{strokeWidth:2.2});
  line([cx-19,cy-2],[cx+5,cy-2],C.blue,2);
  line([cx-19,cy+9],[cx+8,cy+9],C.blue,2);
}
function reader(cx,cy) {
  circle(cx,cy,C.violet,C.paleViolet);
  path("line",[[cx-27,cy-20],[cx-9,cy-24],[cx,cy-18],
    [cx+9,cy-24],[cx+27,cy-20],[cx+27,cy+20],
    [cx+9,cy+16],[cx,cy+21],[cx-9,cy+16],
    [cx-27,cy+20],[cx-27,cy-20]],C.violet,2.5);
  line([cx,cy-18],[cx,cy+21],C.violet,2);
}
function gate(cx,cy) {
  circle(cx,cy,C.amber,C.paleAmber);
  path("line",[[cx,cy-29],[cx+27,cy-15],[cx+21,cy+17],
    [cx,cy+32],[cx-21,cy+17],[cx-27,cy-15],[cx,cy-29]],C.amber,2.5);
  path("line",[[cx-13,cy+1],[cx-3,cy+11],[cx+15,cy-9]],C.amber,2.5);
}
function union(cx,cy) {
  circle(cx,cy,C.coral,C.paleCoral);
  line([cx-20,cy-13],[cx+20,cy-13],C.coral,2);
  line([cx-20,cy-13],[cx,cy+24],C.coral,2);
  line([cx+20,cy-13],[cx,cy+24],C.coral,2);
  for(const p of [[cx-20,cy-13],[cx+20,cy-13],[cx,cy+24]])
    shape("ellipse",p[0]-5,p[1]-5,10,10,C.coral,C.coral,{strokeWidth:1});
}
function final(cx,cy) {
  circle(cx,cy,C.green,C.paleGreen);
  path("line",[[cx-28,cy],[cx-9,cy+18],[cx+31,cy-24]],C.green,4);
}
function under(cx,y,title,detail,w=260) {
  text(title,cx-w/2,y,w,42,27);
  text(detail,cx-w/2-10,y+48,w+20,34,19,C.muted);
}

// Header and unobtrusive domain rules.
text("CARE",50,20,170,66,50,C.ink,"left");
text("Conditional evidence repair over conversation history",
  230,34,1420,51,31,C.muted,"left");
text("SOURCE RECORD",65,127,270,37,23,C.teal,"left");
text("PASS 1  /  COMPACT EVIDENCE",365,127,610,37,23,C.blue,"left");
text("ANSWER CONTROL",1110,127,520,37,23,C.amber,"left");
line([65,173],[285,173],C.teal);
line([365,173],[1040,173],C.blue);
line([1110,173],[1720,173],C.amber);

// All arrows are drawn behind the modules.
for(const row of [[150,420,690,960,1230],[420,690,960,1230]]) {
  const y=row[0]===150?305:735;
  for(let i=0;i<row.length-1;i++)
    path("arrow",[[row[i]+53,y],[row[i]+133,y-12],
      [row[i+1]-53,y]],C.ink,3);
}
path("arrow",[[258,495],[300,445],[300,326],[366,305]],C.blue,3);
path("arrow",[[1282,305],[1418,305],[1513,380],[1559,474]],C.green,3.4);
path("line",[[1270,347],[1375,370],[1375,526]],C.coral,3.3);
line([320,526],[592,526],C.coral,3.3);
line([1060,526],[1375,526],C.coral,3.3);
line([320,526],[320,712],C.coral,3.3);
path("arrow",[[320,712],[367,735]],C.coral,3.3);
path("arrow",[[1282,735],[1445,728],[1480,509],[1549,509]],C.green,3.4);

// First pass.
source(150,305); rank(420,305); bundle(690,305);
reader(960,305); gate(1230,305); final(1600,509);
under(150,375,"Original turns","ID / date / speaker / text");
under(420,375,"Sparse ranker","BM25 / eight turns");
under(690,375,"Visible bundle B1","up to 10k source chars");
under(960,375,"Reader R","answer / IDs / searches");
under(1230,375,"Grounding gate","visible ID / person / event");
under(1600,583,"Final response","answer or abstention",270);
pill("Question q",75,470,185,51,C.blue,C.paleBlue,24);
pill("stop",1421,271,86,34,C.green,C.paper,18);
pill("answer a1",784,195,122,38,C.violet,C.paper,18);
pill("sources C1",916,195,126,38,C.violet,C.paper,18);
pill("searches Q1",1052,195,134,38,C.violet,C.paper,18);
pill("IDs in B",1210,195,122,38,C.amber,C.paper,18);
pill("speaker",1342,195,110,38,C.amber,C.paper,18);
pill("event / time",1462,195,150,38,C.amber,C.paper,18);

// The trigger is a single, explicit condition.
pill("invalid  OR  abstain + (searches OR risk)",
  596,500,460,50,C.coral,C.paper,20);

// Conditional second pass, including all actual retrieval components.
text("PASS 2  /  CONDITIONAL EVIDENCE REPAIR",
  350,562,730,43,24,C.coral,"left");
line([350,606],[1320,606],C.coral);
pill("lexical top 18",348,622,170,38,C.coral,C.paper,18);
pill("stem top 18",532,622,157,38,C.coral,C.paper,18);
pill("neighbors",703,622,135,38,C.coral,C.paper,18);
pill("up to 2 searches",852,622,192,38,C.coral,C.paper,18);
pill("top 6 per search / view",1058,622,254,38,C.coral,C.paper,18);
union(420,735); bundle(690,735); reader(960,735); gate(1230,735);
under(420,803,"Candidate union","same original turns S");
under(690,803,"Visible bundle B2","same 10k-character cap");
under(960,803,"Reader R","one corrective re-read");
under(1230,803,"Grounding gate","same checks / then stop");
text("Original turns remain unchanged. Reference answers and source annotations enter only evaluation.",
  260,930,1350,35,19,C.muted);

const scene={
  type:"excalidraw",version:2,source:"https://excalidraw.com",
  elements,
  appState:{viewBackgroundColor:C.paper,currentItemFontFamily:5,gridSize:null},
  files:{},
};
if(elements.some(e=>e.type==="text"&&e.fontFamily!==5))
  throw new Error("Every label must use Excalifont");
writeFileSync(output,JSON.stringify(scene,null,2)+"\n","utf8");
console.log("Wrote "+output+" ("+elements.length+" editable elements)");
