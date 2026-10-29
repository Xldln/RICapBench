// Official SPICE parsing and semantic equivalence; OA facts use one-to-one matching.
import edu.anu.spice.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.io.Reader;
import java.util.*;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

class SpiceOAScorer {
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

    @SuppressWarnings("unchecked")
    static JSONObject counts(TupleSet predicted, TupleSet reference) {
        boolean[][] allowed = new boolean[predicted.size()][reference.size()];
        for (int p = 0; p < predicted.size(); p++)
            for (int g = 0; g < reference.size(); g++)
                allowed[p][g] = ((SemanticTuple) predicted.get(p)).similarTo((SemanticTuple) reference.get(g));
        int[] owner = new int[reference.size()];
        Arrays.fill(owner, -1);
        int tp = 0;
        for (int p = 0; p < predicted.size(); p++)
            if (augment(p, allowed, owner, new boolean[reference.size()])) tp++;
        JSONObject result = new JSONObject();
        result.put("tp", tp);
        result.put("fp", predicted.size() - tp);
        result.put("fn", reference.size() - tp);
        return result;
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        if (args.length == 1 && args[0].equals("--self-test")) {
            TupleSet p = new TupleSet(new SceneGraph(null)), g = new TupleSet(new SceneGraph(null));
            SemanticConcept car = new SemanticConcept("car", Collections.singleton(1));
            SemanticConcept red = new SemanticConcept("red", Collections.singleton(2));
            for (String word : Arrays.asList("car", "automobile", "vehicle"))
                p.add(new SemanticTuple(new SemanticConcept(word, Collections.singleton(1)), red));
            g.add(new SemanticTuple(car, red));
            g.add(new SemanticTuple(car, new SemanticConcept("blue", Collections.singleton(3))));
            Evaluation official = new Evaluation(p, g, true, true);
            if (official.tp != 3 || official.fn != -1) throw new AssertionError("Negative FN reproduction failed");
            JSONObject fixed = counts(p, g);
            if (!fixed.get("tp").equals(1) || !fixed.get("fp").equals(2) || !fixed.get("fn").equals(1))
                throw new AssertionError("Reference fact was reused: " + fixed);
            TupleSet broad = new TupleSet(new SceneGraph(null)), targets = new TupleSet(new SceneGraph(null));
            SemanticConcept both = new SemanticConcept("a", Collections.emptySet());
            both.merge(new SemanticConcept("b", Collections.emptySet()));
            broad.add(new SemanticTuple(both));
            broad.add(new SemanticTuple(new SemanticConcept("a", Collections.emptySet())));
            targets.add(new SemanticTuple(new SemanticConcept("a", Collections.emptySet())));
            targets.add(new SemanticTuple(new SemanticConcept("b", Collections.emptySet())));
            if (!counts(broad, targets).get("tp").equals(2)) throw new AssertionError("Matching is not maximum");
            System.out.println("Official negative FN reproduced; one-to-one and augmenting-path checks passed");
            return;
        }
        JSONArray input;
        try (Reader reader = Files.newBufferedReader(Paths.get(args[0]), StandardCharsets.UTF_8)) {
            input = (JSONArray) new JSONParser().parse(reader);
        }
        List<String> tests = new ArrayList<>(), refs = new ArrayList<>();
        List<Integer> chunks = new ArrayList<>();
        for (Object item : input) {
            JSONObject row = (JSONObject) item;
            tests.add((String) row.get("test"));
            JSONArray captions = (JSONArray) row.get("refs");
            chunks.add(captions.size());
            for (Object caption : captions) refs.add((String) caption);
        }
        SpiceParser parser = new SpiceParser(args[2], Runtime.getRuntime().availableProcessors(), true);
        List<SceneGraph> references = parser.parseCaptions(refs, chunks);
        List<SceneGraph> predictions = parser.parseCaptions(tests);
        Map<String, TupleFilter> filters = new HashMap<>();
        filters.put("Object", TupleFilter.objectFilter);
        filters.put("Attribute", TupleFilter.attributeFilter);
        filters.put("Relation", TupleFilter.relationFilter);
        filters.put("Color", TupleFilter.colorFilter);
        filters.put("Size", TupleFilter.sizeFilter);
        filters.put("Cardinality", TupleFilter.cardinalityFilter);
        SpiceStats stats = new SpiceStats(filters, false);
        for (int i = 0; i < input.size(); i++)
            stats.score(((JSONObject) input.get(i)).get("image_id"), predictions.get(i), references.get(i), true);
        JSONArray output = stats.toJSONArray();
        for (int i = 0; i < input.size(); i++) {
            JSONObject scores = (JSONObject) ((JSONObject) output.get(i)).get("scores");
            for (String name : Arrays.asList("Object", "Attribute")) {
                TupleFilter filter = filters.get(name);
                scores.put(name, counts(new TupleSet(predictions.get(i), filter), new TupleSet(references.get(i), filter)));
            }
        }
        Files.writeString(Paths.get(args[1]), output.toJSONString(), StandardCharsets.UTF_8);
    }
}
