// Match already extracted spatial triples. Parsing lives in the independent spatial_relations package.
import edu.anu.spice.SemanticConcept;
import edu.anu.spice.SemanticTuple;
import edu.cmu.meteor.aligner.SynonymDictionary;
import edu.cmu.meteor.util.Constants;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.HashMap;
import java.util.Comparator;
import java.util.Set;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

class RICRelationScorer {
    static final Map<String,String> inverses=new HashMap<>();
    static final Map<String,String> aliases=new HashMap<>();
    static final Set<String> symmetric=new HashSet<>();
    static void loadRules(String path) throws Exception {
        inverses.clear(); symmetric.clear(); aliases.clear();
        for(String line:Files.readAllLines(Paths.get(path),StandardCharsets.UTF_8)) {
            if(line.startsWith("#") || line.trim().isEmpty()) continue;
            String[] parts=line.split("\\t");
            if(parts[0].equals("inverse") && parts.length==3) {
                inverses.put(parts[1],parts[2]); inverses.put(parts[2],parts[1]);
            } else if(parts[0].equals("alias") && parts.length==3) aliases.put(parts[1],parts[2]);
            else if(parts[0].equals("symmetric") && parts.length==2) symmetric.add(parts[1]);
            else throw new IllegalArgumentException("Invalid fact equivalence rule: "+line);
        }
    }

    static SemanticConcept concept(String text, SynonymDictionary dictionary) {
        text=text.trim().toLowerCase(Locale.ROOT);
        if(text.contains(" & ")) return new SemanticConcept(text, new HashSet<>());
        SemanticConcept combined=null;
        for(String alias:text.split("/")) {
            String word=alias.trim().replace(" ", "_");
            Set<Integer> synsets=new HashSet<>(dictionary.getSynSets(word));
            synsets.addAll(dictionary.getStemSynSets(word));
            SemanticConcept current=new SemanticConcept(alias.trim(),synsets);
            if(combined==null) combined=current;
            else combined.merge(current);
        }
        return combined;
    }

    static boolean entityMatches(SemanticConcept left, SemanticConcept right, SynonymDictionary dictionary) {
        if(!left.toString().contains(" & ") && !right.toString().contains(" & ")) return left.similarTo(right);
        String[] a=left.toString().split(" & "), b=right.toString().split(" & ");
        if(a.length!=b.length) return false;
        boolean[][] allowed=new boolean[a.length][b.length];
        for(int i=0;i<a.length;i++) for(int j=0;j<b.length;j++)
            allowed[i][j]=concept(a[i],dictionary).similarTo(concept(b[j],dictionary));
        int[] owner=new int[b.length]; Arrays.fill(owner,-1);
        for(int i=0;i<a.length;i++) if(!augment(i,allowed,owner,new boolean[b.length])) return false;
        return true;
    }

    static List<SemanticTuple> references(JSONArray triples, SynonymDictionary dictionary) {
        List<SemanticTuple> result = new ArrayList<>();
        for (Object item : triples) {
            JSONArray triple = (JSONArray) item;
            result.add(new SemanticTuple(
                concept((String) triple.get(0), dictionary),
                new SemanticConcept(aliases.getOrDefault((String)triple.get(1),(String)triple.get(1)), new HashSet<>()),
                concept((String) triple.get(2), dictionary)));
        }
        return result;
    }

    static boolean augment(int p, boolean[][] allowed, int[] owner, boolean[] seen) {
        for (int g = 0; g < owner.length; g++) {
            if (allowed[p][g] && !seen[g]) {
                seen[g] = true;
                if (owner[g] < 0 || augment(owner[g], allowed, owner, seen)) {
                    owner[g] = p;
                    return true;
                }
            }
        }
        return false;
    }

    static boolean equivalent(SemanticTuple a, SemanticTuple b, SynonymDictionary dictionary) {
        String p=a.get(1).toString(), q=b.get(1).toString();
        if(p.equals(q) && entityMatches(a.get(0),b.get(0),dictionary)
                && entityMatches(a.get(2),b.get(2),dictionary)) return true;
        return ((p.equals(q) && symmetric.contains(p)) || q.equals(inverses.get(p)))
            && entityMatches(a.get(0),b.get(2),dictionary)
            && entityMatches(a.get(2),b.get(0),dictionary);
    }

    @SuppressWarnings("unchecked")
    static JSONArray deduplicate(JSONArray triples, SynonymDictionary dictionary) {
        JSONArray sorted=new JSONArray(); sorted.addAll(triples);
        sorted.sort(Comparator.comparing(Object::toString));
        List<SemanticTuple> candidates=references(sorted,dictionary), kept=new ArrayList<>();
        JSONArray output=new JSONArray();
        // ponytail: synonyms are not transitive; compare directly with retained representatives, never close chains.
        for(int i=0;i<candidates.size();i++) {
            boolean duplicate=false;
            for(SemanticTuple representative:kept)
                if(equivalent(candidates.get(i),representative,dictionary)) { duplicate=true; break; }
            if(!duplicate) { kept.add(candidates.get(i)); output.add(sorted.get(i)); }
        }
        return output;
    }

    static int intersection(List<SemanticTuple> predicted, List<SemanticTuple> reference, SynonymDictionary dictionary) {
        boolean[][] allowed = new boolean[predicted.size()][reference.size()];
        for (int p = 0; p < predicted.size(); p++) {
            for (int g = 0; g < reference.size(); g++) {
                SemanticTuple a=predicted.get(p), b=reference.get(g);
                allowed[p][g] = equivalent(a,b,dictionary);
            }
        }
        int[] owner = new int[reference.size()];
        Arrays.fill(owner, -1);
        int tp = 0;
        for (int p = 0; p < predicted.size(); p++) {
            if (augment(p, allowed, owner, new boolean[reference.size()])) tp++;
        }
        return tp;
    }

    static SynonymDictionary dictionary() throws Exception {
        URL dir = Constants.DEFAULT_SYN_DIR_URL;
        return new SynonymDictionary(new URL(dir + "/english.exceptions"),
            new URL(dir + "/english.synsets"), new URL(dir + "/english.relations"));
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        if (args[0].equals("--self-check")) {
            loadRules(args.length>1 ? args[1] : "spatial_relations/fact_equivalence.tsv");
            SemanticConcept car = new SemanticConcept("car", new HashSet<>(Arrays.asList(1)));
            SemanticConcept auto = new SemanticConcept("automobile", new HashSet<>(Arrays.asList(1)));
            SemanticConcept near = new SemanticConcept("near", new HashSet<>());
            SemanticConcept road = new SemanticConcept("road", new HashSet<>());
            SemanticTuple a = new SemanticTuple(car, near, road);
            SemanticTuple b = new SemanticTuple(auto, near, road);
            SemanticTuple reversed = new SemanticTuple(road, near, car);
            SemanticTuple broad = new SemanticTuple(
                new SemanticConcept("vehicle", new HashSet<>(Arrays.asList(1, 2))), near, road);
            SemanticTuple bus = new SemanticTuple(
                new SemanticConcept("bus", new HashSet<>(Arrays.asList(2))), near, road);
            if (intersection(Arrays.asList(a, b), Arrays.asList(a), null) != 1
                    || intersection(Arrays.asList(b), Arrays.asList(a), null) != 1
                    || intersection(Arrays.asList(reversed), Arrays.asList(a), null) != 1
                    || intersection(Arrays.asList(a), new ArrayList<>(), null) != 0
                    || intersection(Arrays.asList(broad, a), Arrays.asList(a, bus), null) != 2) {
                throw new AssertionError("Relation semantic matching failed");
            }
            SynonymDictionary dictionary=dictionary();
            if(!entityMatches(concept("car & truck",dictionary),concept("truck & automobile",dictionary),dictionary)
                || entityMatches(concept("car & truck",dictionary),concept("car",dictionary),dictionary)
                || !entityMatches(concept("pier/dock",dictionary),concept("dock/pier",dictionary),dictionary)
                || new SemanticConcept("left",new HashSet<>()).similarTo(new SemanticConcept("right",new HashSet<>())))
                throw new AssertionError("Grouped endpoints, aliases or predicates failed");
            System.out.println("Relation semantic matching, groups, aliases and fact-equivalence rules passed");
            return;
        }
        loadRules(args[2]);
        JSONArray input;
        try (Reader reader = Files.newBufferedReader(Paths.get(args[0]), StandardCharsets.UTF_8)) {
            input = (JSONArray) new JSONParser().parse(reader);
        }
        SynonymDictionary dictionary = dictionary();
        JSONArray output = new JSONArray();
        for (int i = 0; i < input.size(); i++) {
            if(args.length>3 && args[3].equals("--deduplicate")) {
                output.add(deduplicate((JSONArray)input.get(i),dictionary));
                continue;
            }
            List<SemanticTuple> predicted = references(deduplicate(
                (JSONArray) ((JSONObject) input.get(i)).get("predicted_relations"), dictionary), dictionary);
            List<SemanticTuple> reference = references(deduplicate(
                (JSONArray) ((JSONObject) input.get(i)).get("relations"), dictionary), dictionary);
            int tp = intersection(predicted, reference, dictionary);
            JSONObject counts = new JSONObject();
            counts.put("tp", tp);
            counts.put("fp", predicted.size() - tp);
            counts.put("fn", reference.size() - tp);
            output.add(counts);
        }
        Files.write(Paths.get(args[1]), output.toJSONString().getBytes(StandardCharsets.UTF_8));
    }
}
