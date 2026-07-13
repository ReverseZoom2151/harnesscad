from enum import Enum
class State(str,Enum): INIT="init"; CURVE="curve"; LOOP="loop"; FACE="face"; SKETCH="sketch"; EXTRUSION="extrusion"; PAD="pad"; DEAD="dead"
CURVES={"line","arc","circle"}
def allowed(state):
 return {State.INIT:CURVES,State.CURVE:{"curve_end"},State.LOOP:CURVES|{"loop_end"},State.FACE:{"face_end"},State.SKETCH:{"sketch_end"},State.EXTRUSION:{"add","cut","intersect"},State.PAD:{"pad"},State.DEAD:set()}[state]
def transition(state,token):
 if token not in allowed(state):return State.DEAD
 if state is State.INIT:return State.CURVE
 if state is State.CURVE:return State.LOOP
 if state is State.LOOP:return State.CURVE if token in CURVES else State.FACE
 if state is State.FACE:return State.SKETCH
 if state is State.SKETCH:return State.EXTRUSION
 if state is State.EXTRUSION:return State.PAD
 return state
def run(tokens):
 s=State.INIT
 for i,t in enumerate(tokens):
  s=transition(s,t)
  if s is State.DEAD:return s,(f"illegal:{i}:{t}",)
 return s,()
