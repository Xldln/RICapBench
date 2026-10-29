// Category-anchored spatial phrases with an explicit physical-landmark vocabulary.
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class SpatialRelationParser {
    static class Phrase {
        int start, end;
        String predicate;
        Phrase(int start, int end, String predicate) {
            this.start=start; this.end=end; this.predicate=predicate;
        }
    }
    static class Entity {
        String noun;
        int end;
        Entity(String noun, int end) { this.noun=noun; this.end=end; }
    }
    final StanfordCoreNLP pipeline;
    final Set<String> landmarks = new HashSet<>();
    final Map<String,String> predicates = new HashMap<>();
    final Map<String,String> inverses = new HashMap<>(), aliases = new HashMap<>();
    final Pattern phrases;
    static final Pattern AND = Pattern.compile("(?<![a-z-])and(?![a-z-])");
    static final Pattern NEGATION = Pattern.compile("\\b(?:not|never|neither|without|no)\\b");
    static final Pattern ATTRIBUTE = Pattern.compile("\\b(?:with|without|has|have|having|features?|featuring|carrying|wearing|shows?|showing|contains?|includes?|including|obscured by|covered by|shaded by|hidden by)\\b");
    static final Pattern ALTERNATIVE = Pattern.compile("\\b(?:or|either)\\b");
    static final Pattern RELATIVE = Pattern.compile("\\b(?:who|which|that|whose|where|while|whereas)\\b");
    static final Pattern ELLIPSIS = Pattern.compile(
        "(?:(?:a|an|the|another|other|dark|light|lighter|darker|white|black|blue|red|yellow|green|gray|grey|small|large)\\s*)+");
    static final String DIRECTION = "(?:upper|lower|above|below)[ -](?:left|right)|left|right|above|below|ahead|behind|front|north|south|east|west";
    static final String REFERENCE_DIRECTION_TEXT =
        "\\s*(?:(?:just|directly|immediately|parked|positioned|located|standing|sitting)\\s+)*"
        + "(?:(?:to|on|at)\\s+(its|their)\\s+|in\\s+(?=front\\s+of\\s+(?:it|them)\\b))?("+DIRECTION+")\\b(?:\\s+side)?(?:\\s+(?:of\\s+)?(it|them))?";
    static final Pattern REFERENCE_DIRECTION = Pattern.compile("^"+REFERENCE_DIRECTION_TEXT);
    static final Pattern MORE_DIRECTION = Pattern.compile(
        "^\\s*(?:,\\s*(?:and\\s+)?|and\\s+)");
    static final Pattern DIRECTION_END = Pattern.compile("^\\s*(?:$|[,;.!?]|(?:and|but|or)\\b)");
    static final Set<String> FORBIDDEN = new HashSet<>(Arrays.asList(
        "area", "screen", "scene", "image", "picture", "frame"));

    SpatialRelationParser(String rules, int threads, String objects, String facts) throws Exception {
        for(String line:Files.readAllLines(Paths.get(objects),StandardCharsets.UTF_8))
            if(!line.startsWith("#") && !line.trim().isEmpty()) landmarks.add(line.trim());
        for(String line:Files.readAllLines(Paths.get(rules),StandardCharsets.UTF_8)) {
            if(line.startsWith("#") || line.trim().isEmpty()) continue;
            String[] parts=line.split("\\t",2);
            for(String alias:parts[1].split("\\|")) predicates.put(alias,parts[0]);
        }
        for(String line:Files.readAllLines(Paths.get(facts),StandardCharsets.UTF_8)) {
            if(line.startsWith("#") || line.trim().isEmpty()) continue;
            String[] parts=line.split("\\t");
            if(parts[0].equals("inverse")) {
                inverses.put(parts[1],parts[2]); inverses.put(parts[2],parts[1]);
            } else if(parts[0].equals("alias")) aliases.put(parts[1],parts[2]);
        }
        // "nearby" is often an adjective inside another relation's object phrase.
        predicates.remove("nearby");
        // Participial forms retain the annotated instance as subject without reversing endpoints.
        predicates.put("surrounded by","surrounded_by");
        predicates.put("encircled by","surrounded_by");
        predicates.put("bordered by","bordered_by");
        predicates.put("flanked by","bordered_by");
        for(String verb:Arrays.asList("surround","border","span","intersect","overlap","touch"))
            predicates.put(verb+"s",verb);
        predicates.put("crosses","intersect");
        predicates.put("straddles","span");
        predicates.put("adjoins","border");
        predicates.put("abuts","border");
        for(String direction:Arrays.asList("left", "right", "upper-left", "upper-right", "lower-left", "lower-right",
                                           "above-left", "above-right", "below-left", "below-right")) {
            for(String spelling:Arrays.asList(direction,direction.replace('-', ' '))) {
                for(String prep:Arrays.asList("in", "on", "at", "along", "to", "occupies"))
                    for(String article:Arrays.asList(" ", " the ", " a "))
                        for(String suffix:Arrays.asList("", " of", " side of", " corner of", " edge of", " part of", " area of"))
                            predicates.put(prep+article+spelling+suffix,direction);
                for(String suffix:Arrays.asList(" of", " side of", " corner of", " edge of"))
                    predicates.put(spelling+suffix,direction);
            }
        }
        List<String> aliases=new ArrayList<>(predicates.keySet());
        // Bare directional adjectives inside an object NP must not start a new predicate.
        aliases.removeAll(Arrays.asList("left", "right", "adjacent", "parallel", "perpendicular",
                                       "close", "north", "south", "east", "west"));
        aliases.sort(Comparator.comparingInt(String::length).reversed().thenComparing(x->x));
        List<String> patterns=new ArrayList<>();
        for(String alias:aliases) patterns.add(Pattern.quote(alias));
        phrases=Pattern.compile("(?<![a-z-])(?:"+String.join("|",patterns)+")(?![a-z-])");
        Properties props=new Properties();
        // Subject/category and phrase scope are explicit; POS errors cannot invent entity names.
        props.setProperty("annotators","tokenize,ssplit,pos,lemma");
        props.setProperty("threads",Integer.toString(threads));
        pipeline=new StanfordCoreNLP(props);
    }

    static final Set<String> MODIFIERS = new HashSet<>(Arrays.asList(
        "cargo", "storage", "tennis", "basketball", "baseball", "football", "soccer", "running",
        "traffic", "service", "toll", "passenger", "fishing", "engineering", "liquid", "dry",
        "white", "black", "blue", "red", "yellow", "green", "gray", "grey", "tan", "dark", "light",
        "silver", "gold", "brown", "orange", "pink", "purple", "beige", "teal", "magenta",
        "brightly", "diagonally", "vertically", "horizontally", "lit", "paved", "parked", "parking"));
    static final Set<String> NON_ENTITIES = new HashSet<>(Arrays.asList(
        "row", "group", "cluster", "column", "formation", "orientation", "view", "perspective", "direction",
        "clothing", "pants", "shirt", "attire", "night", "day", "daylight", "illumination", "shadow"));

    static final Set<String> GROUPS = new HashSet<>(Arrays.asList(
        "row", "group", "cluster", "column", "line", "fleet", "crowd", "formation", "collection", "queue"));

    String nounForm(CoreLabel token) {
        return token.tag().equals("NNS") || token.tag().equals("NNPS")
            ? token.word().toLowerCase(Locale.ROOT) : token.lemma().toLowerCase(Locale.ROOT);
    }

    boolean nounToken(CoreLabel token, String text, int start) {
        if(token.tag().startsWith("NN")) return true;
        if(!landmarks.contains(token.lemma().toLowerCase(Locale.ROOT))) return false;
        String prefix=text.substring(start,token.beginPosition()).trim();
        // CoreNLP tags "a pedestrian" as JJ and "a market stall" as NN VB.
        return (token.tag().equals("JJ") && prefix.matches("a|an|the|another|other|one"))
            || (token.tag().equals("VB") && prefix.matches("(?:a|an|the)\\s+[a-z-]+"));
    }

    Entity object(List<CoreLabel> tokens, String text, int start, int end) {
        for(CoreLabel token:tokens) {
            if(token.beginPosition()<start || token.endPosition()>end) continue;
            String word=token.word().toLowerCase(Locale.ROOT), lemma=token.lemma().toLowerCase(Locale.ROOT);
            if(lemma.equals("people")) lemma="person";
            // An excluded head must not be bypassed to pick a noun in a later relative clause.
            if(RELATIVE.matcher(word).matches() || ATTRIBUTE.matcher(word).matches() || FORBIDDEN.contains(lemma)) return null;
            if(GROUPS.contains(lemma)) {
                Matcher of=Pattern.compile("\\s+of\\s+").matcher(text.substring(token.endPosition(),end));
                if(!of.lookingAt()) return null;
                return object(tokens,text,token.endPosition()+of.end(),end);
            }
            if(NON_ENTITIES.contains(lemma) || Arrays.asList("VBZ", "VBP", "VBD", "MD").contains(token.tag())) return null;
            boolean trafficHead=lemma.matches("sign|light")
                && text.substring(start,token.beginPosition()).matches("(?s).*\\btraffic\\s+$");
            if((nounToken(token,text,start) || trafficHead)
                    && landmarks.contains(trafficHead ? "traffic "+lemma : lemma) && !predicates.containsKey(lemma)) {
                Entity candidate=new Entity((trafficHead ? "traffic " : "")+nounForm(token),token.endPosition());
                // Keep the last head of a contiguous physical compound: water channel, cargo ship.
                for(CoreLabel next:tokens) {
                    if(next.beginPosition()<candidate.end || next.endPosition()>end) continue;
                    if(!text.substring(candidate.end,next.beginPosition()).trim().isEmpty()) break;
                    String nextLemma=next.lemma().toLowerCase(Locale.ROOT);
                    if(FORBIDDEN.contains(nextLemma)
                            || (next.tag().startsWith("NN") && NON_ENTITIES.contains(nextLemma))) return null;
                    if(next.word().toLowerCase(Locale.ROOT).matches(DIRECTION)) break;
                    boolean nextTraffic=nextLemma.matches("sign|light")
                        && text.substring(start,next.beginPosition()).matches("(?s).*\\btraffic\\s+$");
                    if(!nounToken(next,text,start) && !nextTraffic) {
                        if(landmarks.contains(nextLemma) && hasClauseVerb(tokens,next.beginPosition(),next.endPosition())) return null;
                        break;
                    }
                    // The final noun is the head: "road markings" are markings, not a road.
                    if(Arrays.asList("side", "edge", "corner", "end").contains(nextLemma)) {
                        candidate.end=next.endPosition(); continue;
                    }
                    if(nextTraffic) {
                        candidate=new Entity("traffic "+nounForm(next),next.endPosition()); continue;
                    }
                    candidate=new Entity(landmarks.contains(nextLemma) && !predicates.containsKey(nextLemma)
                            ? nounForm(next) : null,next.endPosition());
                }
                return candidate.noun==null ? null : candidate;
            }
            if(token.tag().startsWith("NN") && !MODIFIERS.contains(lemma)
                    && !word.matches("(?:upper|lower|top|bottom|left|right)(?:-(?:left|right))?")
                    && !Arrays.asList("side", "edge", "corner", "end").contains(lemma)) {
                // An unknown noun may modify an adjacent head, never a noun across a PP/clause.
                boolean modifier=false;
                for(CoreLabel next:tokens) {
                    if(next.beginPosition()<token.endPosition() || next.endPosition()>end) continue;
                    modifier=text.substring(token.endPosition(),next.beginPosition()).trim().isEmpty()
                        && nounToken(next,text,start);
                    break;
                }
                if(!modifier) return null;
            }
        }
        return null;
    }

    boolean hasClauseVerb(List<CoreLabel> tokens, int start, int end) {
        for(CoreLabel token:tokens)
            if(token.beginPosition()>=start && token.endPosition()<=end
                    && (Arrays.asList("VBZ", "VBP", "VBD", "MD").contains(token.tag())
                        || token.word().matches("sits|stands|lies|rests|floats|extends|drives|travels|walks|rides"))) return true;
        return false;
    }

    int attributeScope(List<CoreLabel> tokens, String text, int start, int end) {
        Matcher attributes=ATTRIBUTE.matcher(text);
        attributes.region(start,end);
        int scope=start;
        while(attributes.find()) {
            // "parallel with" is a spatial phrase, not an attribute attachment.
            if(text.substring(start,attributes.start()).matches("(?s).*\\bparallel\\s+$")) continue;
            scope=-1;
            for(CoreLabel token:tokens) {
                if(token.beginPosition()<attributes.end() || token.endPosition()>end) continue;
                if(Arrays.asList("VBZ", "VBP", "VBD", "MD").contains(token.tag())
                        || token.word().matches("sits|stands|lies|rests|floats|extends|drives|travels|walks|rides")) {
                    if(!RELATIVE.matcher(text.substring(attributes.end(),token.beginPosition())).find())
                        scope=token.endPosition();
                    break;
                }
            }
            int comma=text.lastIndexOf(',',end);
            Matcher continuation=Pattern.compile("\\band\\s+(?=positioned|located|situated|oriented|parked|moored|docked|aligned)").matcher(text);
            continuation.region(attributes.end(),end);
            while(continuation.find()) comma=Math.max(comma,continuation.end()-1);
            String prefix=comma>=attributes.end() ? text.substring(comma+1,end).trim() : "!";
            Matcher spatial=phrases.matcher(prefix);
            if(spatial.find()) prefix=prefix.substring(0,spatial.start()).trim();
            if(comma>=attributes.end() && prefix.replaceFirst("^(?:just|immediately|directly)\\s*", "").matches(
                    "(?:(?:and|also)\\s+)?(?:(?:just|immediately|directly)\\s+)?(?:(?:parked|positioned|located|situated|oriented|moored|docked|aligned|standing|sitting|resting|running|nestled|set)(?:\\s+(?:just|immediately|centrally|diagonally|horizontally|vertically|directly|bow-up))*\\s*)?"))
                scope=comma+1;
        }
        return scope;
    }

    boolean referenceDirections(Set<List<String>> result, String subject, Entity reference,
                             String text, int end, boolean implicit) {
        String tail=text.substring(reference.end,end);
        Matcher direction=REFERENCE_DIRECTION.matcher(tail);
        if(!direction.find() || (!implicit && direction.group(1)==null && direction.group(3)==null)) return false;
        List<String> directions=new ArrayList<>();
        int consumed=0, offset=0;
        boolean comma=false;
        while(true) {
            int nextEnd=offset+direction.end();
            // A direction with its own complement does not use the target as its frame.
            if(!DIRECTION_END.matcher(tail.substring(nextEnd)).find()) {
                if(comma) break;
                return false;
            }
            directions.add(direction.group(2).replace(' ','-'));
            consumed=nextEnd;
            Matcher separator=MORE_DIRECTION.matcher(tail.substring(consumed));
            if(!separator.find()) break;
            comma=separator.group().contains(",");
            offset=consumed+separator.end();
            direction=REFERENCE_DIRECTION.matcher(tail.substring(offset));
            if(!direction.find()) break;
        }
        if(ALTERNATIVE.matcher(tail.substring(0,consumed)).find()
                || tail.substring(consumed).matches("^\\s+or\\b.*")) return false;
        List<String> canonical=new ArrayList<>();
        for(String value:directions) {
            if(value.equals("ahead")) value="front";
            canonical.add(aliases.getOrDefault(value,value));
        }
        for(String value:canonical)
            if(canonical.contains(inverses.get(value))) return false;
        for(String value:canonical) {
            String inverse=inverses.get(value);
            if(inverse!=null) add(result,subject,inverse,Arrays.asList(reference));
        }
        return true;
    }

    void withReferences(Set<List<String>> result, List<CoreLabel> tokens, String text,
                        String subject, int start, int end) {
        Matcher with=Pattern.compile("\\b(?:with|has|having)\\s+").matcher(text);
        with.region(start,end);
        while(with.find()) {
            int clause=Math.max(start,text.lastIndexOf(',',with.start())+1);
            String prefix=text.substring(clause,with.start()).trim();
            // A comma-led reference clause or a direct target attribute has an explicit owner.
            if(!prefix.matches("(?:(?:and|is|are|was|were|parked|positioned|located|situated|moored|docked|stands|sits|just|directly|vertically|horizontally|diagonally)\\s*)*")
                    || attributeScope(tokens,text,start,with.start())<0) continue;
            int stop=end;
            Matcher boundary=Pattern.compile(",(?!\\s*(?:and\\s+)?"+REFERENCE_DIRECTION_TEXT+")\\s*|\\band\\s+(?=(?:a|an|the|another|other|several|many|[0-9]+)\\b)").matcher(text);
            boundary.region(with.end(),end);
            List<int[]> parts=new ArrayList<>();
            int cursor=with.end();
            while(boundary.find()) { parts.add(new int[]{cursor,boundary.start()}); cursor=boundary.end(); }
            parts.add(new int[]{cursor,stop});
            for(int[] part:parts) {
                String span=text.substring(part[0],part[1]);
                if(NEGATION.matcher(span).find() || RELATIVE.matcher(span).find()
                        || ALTERNATIVE.matcher(span).find()) continue;
                if(hasClauseVerb(tokens,part[0],part[1])) break;
                Entity reference=object(tokens,text,part[0],part[1]);
                if(reference==null || hasClauseVerb(tokens,reference.end,part[1])) continue;
                String before=text.substring(part[0],reference.end);
                String after=text.substring(reference.end,part[1]);
                boolean nearby=before.matches("(?s).*\\bnearby\\b.*") || after.matches("^\\s+nearby\\b.*");
                if(!referenceDirections(result,subject,reference,text,part[1],true) && !nearby) break;
                if(nearby)
                    add(result,subject,"near",Arrays.asList(reference));
            }
        }
    }

    void add(Set<List<String>> result, String subject, String predicate, List<Entity> objects) {
        if(objects.isEmpty()) return;
        predicate=aliases.getOrDefault(predicate,predicate);
        if(predicate.equals("between")) {
            List<String> members=new ArrayList<>();
            for(Entity object:objects) members.add(object.noun);
            Collections.sort(members);
            result.add(Arrays.asList(subject,predicate,String.join(" & ",members)));
        } else for(Entity object:objects) result.add(Arrays.asList(subject,predicate,object.noun));
    }

    List<List<String>> extract(Annotation annotation, String subject, String surface, int start) {
        Set<List<String>> result=new LinkedHashSet<>();
        String text=annotation.get(CoreAnnotations.TextAnnotation.class).toLowerCase(Locale.ROOT);
        List<CoreMap> sentences=annotation.get(CoreAnnotations.SentencesAnnotation.class);
        if(sentences.isEmpty()) return new ArrayList<>();
        List<CoreLabel> tokens=sentences.get(0).get(CoreAnnotations.TokensAnnotation.class);
        int end=tokens.get(tokens.size()-1).endPosition();
        // A category after a finite verb is an object, even if the opening subject is unknown.
        int opening=Math.max(0,text.lastIndexOf(',',start)+1);
        if(start+surface.length()>end || hasClauseVerb(tokens,opening,start)) return new ArrayList<>();
        List<Phrase> found=new ArrayList<>();
        Matcher matcher=phrases.matcher(text);
        matcher.region(start+surface.length(),end);
        while(matcher.find()) found.add(new Phrase(matcher.start(),matcher.end(),predicates.get(matcher.group())));
        int lastObjectEnd=start+surface.length();
        boolean previousValid=false;
        String previousPredicate="";
        String previousObject="";
        for(int i=0;i<found.size();i++) {
            Phrase phrase=found.get(i);
            int clauseStart=Math.max(start+surface.length(),text.lastIndexOf(',',phrase.start)+1);
            int scope=attributeScope(tokens,text,start+surface.length(),phrase.start);
            if(scope<0) continue;
            clauseStart=Math.max(clauseStart,scope);
            Matcher contrast=Pattern.compile("\\bbut\\b").matcher(text);
            contrast.region(clauseStart,phrase.start);
            while(contrast.find()) clauseStart=contrast.end();
            String clausePrefix=text.substring(clauseStart,phrase.start).replaceAll("\\bnot only\\b", "");
            boolean firstInClause=i==0 || found.get(i-1).start<clauseStart;
            if(firstInClause && clauseStart > start+surface.length() && clauseStart!=scope
                    && object(tokens,text,clauseStart,phrase.start)!=null) continue;
            Matcher coordination=Pattern.compile("[,;]|\\b(?:and|but)\\b").matcher(text);
            coordination.region(start+surface.length(),phrase.start);
            int coordinatedStart=-1;
            while(coordination.find()) coordinatedStart=coordination.end();
            // A new explicit entity owns its following predicates, even if our preceding fact was rejected.
            if(coordinatedStart>=scope && object(tokens,text,coordinatedStart,phrase.start)!=null
                    && hasClauseVerb(tokens,coordinatedStart,phrase.start)) break;
            if(NEGATION.matcher(clausePrefix).find() || RELATIVE.matcher(clausePrefix).find()
                    || ALTERNATIVE.matcher(clausePrefix).find()) continue;
            if(previousValid) {
                String transition=text.substring(Math.min(lastObjectEnd,phrase.start),phrase.start);
                int transitionStart=Math.max(lastObjectEnd,clauseStart);
                if(hasClauseVerb(tokens,transitionStart,phrase.start)
                        && object(tokens,text,transitionStart,phrase.start)!=null) continue;
                // An unseparated second PP can describe the reference object. Keep only explicit coordination.
                boolean peer=previousObject.matches("(?:car|vehicle|truck|van|bus|bicycle|motorcycle|person|pedestrian)(?:s|es)?|people");
                boolean peerLocation=peer && previousPredicate.matches("above|below|beside|near|front|behind|left|right|opposite")
                    && phrase.predicate.matches("in|on|at|along|left|right|upper-left|upper-right|lower-left|lower-right");
                boolean referenceAdjective=previousPredicate.matches("near|beside")
                    && text.substring(phrase.start,phrase.end).matches("adjacent(?: to)?");
                boolean referenceSurface=previousPredicate.equals("between") && phrase.predicate.equals("on");
                if(!transition.contains(",") && !transition.matches("(?s).*\\b(?:and|also|but)\\b.*")
                        && (!transition.trim().isEmpty() || peerLocation || referenceAdjective || referenceSurface)) continue;
            }
            if(phrase.predicate.equals("between")) {
                int stop=text.indexOf(',',phrase.end);
                if(stop<0 || stop>end) stop=end;
                if(!AND.matcher(text.substring(phrase.end,stop)).find() && i+1<found.size())
                    stop=Math.min(stop,found.get(i+1).start);
                String span=text.substring(phrase.end,stop);
                if(NEGATION.matcher(span).find() || ALTERNATIVE.matcher(span).find()) continue;
                List<int[]> parts=new ArrayList<>();
                Matcher conjunction=AND.matcher(text); conjunction.region(phrase.end,stop);
                int cursor=phrase.end;
                while(conjunction.find()) { parts.add(new int[]{cursor,conjunction.start()}); cursor=conjunction.end(); }
                parts.add(new int[]{cursor,stop});
                List<Entity> members=new ArrayList<>();
                boolean incomplete=false;
                for(int n=0;n<parts.size();n++) {
                    int[] part=parts.get(n);
                    String fragment=text.substring(part[0],part[1]).trim();
                    Matcher following=phrases.matcher(fragment);
                    if(n>0 && following.lookingAt()) { stop=part[0]; break; }
                    Entity member=object(tokens,text,part[0],part[1]);
                    if(member==null && ELLIPSIS.matcher(fragment).matches() && n+1<parts.size()) {
                        Entity shared=object(tokens,text,parts.get(n+1)[0],parts.get(n+1)[1]);
                        if(shared!=null) member=new Entity(shared.noun,part[1]);
                    }
                    if(member==null || hasClauseVerb(tokens,member.end,part[1])) { incomplete=true; break; }
                    members.add(member);
                }
                boolean plural=false;
                if(members.size()==1)
                    for(CoreLabel token:tokens)
                        if((token.endPosition()==members.get(0).end && token.tag().startsWith("NN") && token.tag().endsWith("S"))
                                || (token.beginPosition()>=phrase.end && token.endPosition()<=members.get(0).end
                                    && token.tag().equals("CD") && !token.word().matches("one|1|zero|0"))) plural=true;
                if(!incomplete && (members.size()>1 || plural)) {
                    add(result,subject,"between",members);
                    previousValid=true;
                    previousPredicate="between";
                    previousObject=members.get(members.size()-1).noun;
                    lastObjectEnd=members.get(members.size()-1).end;
                }
                if(!incomplete && (members.size()>1 || plural))
                    while(i+1<found.size() && found.get(i+1).start<stop) i++;
                continue;
            }
            List<String> sharedPredicates=new ArrayList<>();
            sharedPredicates.add(phrase.predicate);
            while(i+1<found.size() && text.substring(phrase.end,found.get(i+1).start).trim().equals("and")
                    && !phrase.predicate.equals("between") && !found.get(i+1).predicate.equals("between")) {
                phrase=found.get(++i);
                sharedPredicates.add(phrase.predicate);
            }
            int stop=i+1<found.size() ? found.get(i+1).start : end;
            int comma=text.indexOf(',',phrase.end);
            if(comma>=0) stop=Math.min(stop,comma);
            Matcher negativeContinuation=Pattern.compile("\\b(?:and|but)\\s+(?:not|never|neither|no)\\b").matcher(text);
            negativeContinuation.region(phrase.end,stop);
            if(negativeContinuation.find()) stop=negativeContinuation.start();
            Matcher attribute=ATTRIBUTE.matcher(text);
            attribute.region(phrase.end,stop);
            if(attribute.find()) stop=attribute.start();
            if(stop<=phrase.end) continue;
            String segment=text.substring(phrase.end,stop);
            if(NEGATION.matcher(segment).find() || ALTERNATIVE.matcher(segment).find()) continue;
            // Possessive/postposed direction: "on the person's left" / "in the person left".
            Matcher postposed=Pattern.compile("\\b(upper-left|upper-right|lower-left|lower-right|left|right|front|behind)\\b(?:\\s+side)?[. ]*$").matcher(segment);
            if(phrase.predicate.matches("in|on|at|to") && postposed.find()
                    && object(tokens,text,phrase.end,phrase.end+postposed.start())!=null) {
                phrase=new Phrase(phrase.start,phrase.end,postposed.group(1));
                sharedPredicates.clear(); sharedPredicates.add(phrase.predicate);
                stop=phrase.end+postposed.start();
            }
            // Pure directions and attributes have no allowed physical reference and are dropped.
            List<Entity> objects=new ArrayList<>();
            List<int[]> pieces=new ArrayList<>();
            Matcher and=AND.matcher(text);
            and.region(phrase.end,stop);
            int cursor=phrase.end;
            boolean nextClause=false;
            while(and.find()) {
                pieces.add(new int[]{cursor,and.start()}); cursor=and.end();
                if(hasClauseVerb(tokens,cursor,stop)) { nextClause=true; break; }
            }
            if(!nextClause) pieces.add(new int[]{cursor,stop});
            for(int[] piece:pieces) {
                Entity object=object(tokens,text,piece[0],piece[1]);
                if(object!=null) objects.add(object);
            }
            if(!objects.isEmpty()) {
                for(String predicate:sharedPredicates) add(result,subject,predicate,objects);
                for(Entity reference:objects) referenceDirections(result,subject,reference,text,end,false);
                lastObjectEnd=objects.get(objects.size()-1).end;
                previousValid=true;
                previousPredicate=phrase.predicate;
                previousObject=objects.get(objects.size()-1).noun;
            }
        }
        withReferences(result,tokens,text,subject,start+surface.length(),end);
        List<List<String>> sorted=new ArrayList<>(result);
        sorted.sort(Comparator.comparing(t->String.join("\u0001",t)));
        return sorted;
    }

    static String quote(String text) {
        StringBuilder result=new StringBuilder("\"");
        for(char c:text.toCharArray()) {
            if(c=='"' || c=='\\') result.append('\\').append(c);
            else if(c<32) result.append(String.format("\\u%04x",(int)c));
            else result.append(c);
        }
        return result.append('"').toString();
    }
    static String json(List<List<String>> tuples) {
        List<String> rows=new ArrayList<>();
        for(List<String> tuple:tuples) {
            List<String> words=new ArrayList<>();
            for(String word:tuple) words.add(quote(word));
            rows.add("["+String.join(",",words)+"]");
        }
        return "["+String.join(",",rows)+"]";
    }
    public static void main(String[] args) throws Exception {
        SpatialRelationParser parser=new SpatialRelationParser(args[2],Integer.parseInt(args[3]),args[4],args[5]);
        List<Annotation> annotations=new ArrayList<>();
        List<String[]> requests=new ArrayList<>();
        for(String text:Files.readAllLines(Paths.get(args[0]),StandardCharsets.UTF_8)) {
            String[] request=text.split("\\t",4); requests.add(request);
            annotations.add(new Annotation(request[3]));
        }
        parser.pipeline.annotate(annotations);
        List<String> output=new ArrayList<>();
        for(int i=0;i<annotations.size();i++) {
            String[] request=requests.get(i);
            output.add(json(parser.extract(annotations.get(i),request[0],request[1],Integer.parseInt(request[2]))));
        }
        Files.write(Paths.get(args[1]),output,StandardCharsets.UTF_8);
    }
}
