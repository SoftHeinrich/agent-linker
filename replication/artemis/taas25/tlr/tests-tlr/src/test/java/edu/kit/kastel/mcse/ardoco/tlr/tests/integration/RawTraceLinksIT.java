/* Licensed under MIT 2026. */
package edu.kit.kastel.mcse.ardoco.tlr.tests.integration;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Collection;
import java.util.TreeSet;
import java.util.stream.Stream;

import org.eclipse.collections.api.factory.Sets;
import org.eclipse.collections.api.factory.SortedMaps;
import org.eclipse.collections.api.set.MutableSet;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import edu.kit.kastel.mcse.ardoco.core.api.entity.ModelEntity;
import edu.kit.kastel.mcse.ardoco.core.api.models.Metamodel;
import edu.kit.kastel.mcse.ardoco.core.api.models.ModelFormat;
import edu.kit.kastel.mcse.ardoco.core.api.output.ArDoCoResult;
import edu.kit.kastel.mcse.ardoco.core.api.stage.connectiongenerator.ner.NamedArchitectureEntityOccurrence;
import edu.kit.kastel.mcse.ardoco.core.api.stage.connectiongenerator.ner.NerConnectionState;
import edu.kit.kastel.mcse.ardoco.core.api.tracelink.TraceLink;
import edu.kit.kastel.mcse.ardoco.core.common.util.Environment;
import edu.kit.kastel.mcse.ardoco.core.execution.runner.ArDoCoRunner;
import edu.kit.kastel.mcse.ardoco.tlr.execution.ArtemisInTransArC;
import edu.kit.kastel.mcse.ardoco.tlr.models.agents.ArchitectureConfiguration;
import edu.kit.kastel.mcse.ardoco.tlr.models.agents.CodeConfiguration;
import edu.kit.kastel.mcse.ardoco.tlr.models.informants.LargeLanguageModel;
import edu.kit.kastel.mcse.ardoco.tlr.tests.approach.TransArCEvaluationProject;

class RawTraceLinksIT {
    private static final LargeLanguageModel LLM = selectLlm();
    private static final File OUTPUT_ROOT = new File(new File("target", "raw-tracelinks"), LLM.name());

    private static LargeLanguageModel selectLlm() {
        String name = Environment.getEnv("ARTEMIS_LLM");
        if (name == null || name.isBlank()) {
            return LargeLanguageModel.GPT_5_5;
        }
        return LargeLanguageModel.valueOf(name.trim());
    }

    @BeforeAll
    static void beforeAll() throws IOException {
        Assumptions.assumeTrue(Environment.getEnv("OPENAI_API_KEY") != null,
                "OPENAI_API_KEY must be set for live OpenAI calls");
        Files.createDirectories(OUTPUT_ROOT.toPath());
    }

    @DisplayName("Dump raw SAD-SAM and SAD-Code trace links (ArTEMiS + TransArC)")
    @ParameterizedTest(name = "{0}")
    @MethodSource("projects")
    void dumpTraceLinks(TransArCEvaluationProject project) throws IOException {
        ArDoCoRunner runner = createPipeline(project, LLM);
        ArDoCoResult result = runner.run();
        Assertions.assertNotNull(result);

        writeSadSamLinks(project, result);
        writeSadCodeLinks(project, result);
    }

    private static ArDoCoRunner createPipeline(TransArCEvaluationProject project, LargeLanguageModel llm) {
        String projectName = project.name().toLowerCase();
        File textInput = project.getTlrTask().getTextFile();
        ModelFormat architectureModelFormat = ModelFormat.PCM;
        File architectureModel = project.getTlrTask().getArchitectureModelFile(architectureModelFormat);
        CodeConfiguration codeConfig = new CodeConfiguration(project.getTlrTask().getCodeModelFromResources(),
                CodeConfiguration.CodeConfigurationType.ACM_FILE);
        File outputDir = new File("target", projectName + "-output");
        outputDir.mkdirs();
        ArtemisInTransArC pipeline = new ArtemisInTransArC(projectName);
        pipeline.setUp(textInput, new ArchitectureConfiguration(architectureModel, architectureModelFormat), codeConfig,
                SortedMaps.immutable.empty(), outputDir, llm);
        return pipeline;
    }

    private static void writeSadSamLinks(TransArCEvaluationProject project, ArDoCoResult result) throws IOException {
        File out = new File(OUTPUT_ROOT, project.name() + "-sad-sam.tsv");
        NerConnectionState nerState = result.getNerConnectionState(Metamodel.ARCHITECTURE_WITH_COMPONENTS);
        Collection<TraceLink<NamedArchitectureEntityOccurrence, ModelEntity>> links = nerState.getTraceLinks().castToCollection();
        MutableSet<String> rows = Sets.mutable.empty();
        for (TraceLink<NamedArchitectureEntityOccurrence, ModelEntity> tl : links) {
            int sentence = tl.getFirstEndpoint().getSentenceNumber();
            String sam = tl.getSecondEndpoint().getId();
            rows.add(sentence + "\t" + sam);
        }
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out.toPath(), StandardCharsets.UTF_8))) {
            pw.println("# sad_sentence_index\tsam_entity_id");
            for (String row : new TreeSet<>(rows)) {
                pw.println(row);
            }
        }
    }

    private static void writeSadCodeLinks(TransArCEvaluationProject project, ArDoCoResult result) throws IOException {
        File out = new File(OUTPUT_ROOT, project.name() + "-sad-code.tsv");
        var links = result.getSadCodeTraceLinks();
        MutableSet<String> rows = Sets.mutable.empty();
        links.forEach(tl -> {
            int sentence = tl.getFirstEndpoint().getSentence().getSentenceNumber() + 1;
            String codeId = tl.getSecondEndpoint().toString();
            rows.add(sentence + "\t" + codeId);
        });
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out.toPath(), StandardCharsets.UTF_8))) {
            pw.println("# sad_sentence_index\tcode_endpoint_id");
            for (String row : new TreeSet<>(rows)) {
                pw.println(row);
            }
        }
    }

    private static Stream<Arguments> projects() {
        return Stream.of(TransArCEvaluationProject.values()).map(Arguments::of);
    }
}
