//! The GitHub workflows, read as GitHub reads them before it runs anything:
//! every file is YAML, every job is one GitHub accepts, and the jobs, steps,
//! inputs and matrix entries its expressions name are there. GitHub says so
//! only once a workflow is pushed, and some of it only when the job runs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use yaml_serde::Value;

const CALLED: &str = "./.github/workflows/";

/// What is wrong with these workflow files (name, text), one line each.
fn problems(files: &BTreeMap<String, String>) -> Vec<String> {
    let mut wrong = Vec::new();
    let mut parsed = BTreeMap::new();
    for (name, text) in files {
        match yaml_serde::from_str::<Value>(text) {
            Ok(workflow) => drop(parsed.insert(name.as_str(), workflow)),
            Err(error) => wrong.push(format!("{name}: not YAML: {error}")),
        }
    }
    for (name, workflow) in &parsed {
        let mut say = |problem: String| wrong.push(format!("{name}: {problem}"));
        let Some(jobs) = workflow.get("jobs").and_then(Value::as_mapping) else {
            say("no jobs".into());
            continue;
        };
        if workflow.get("on").is_none() {
            say("no `on`".into());
        }
        let inputs = declared_inputs(workflow);
        let jobs: BTreeMap<&str, &Value> = jobs
            .iter()
            .filter_map(|(key, job)| Some((key.as_str()?, job)))
            .collect();
        // Outside the jobs only the inputs can be checked (`env`, `concurrency`, `run-name`).
        for (key, value) in workflow.as_mapping().into_iter().flatten() {
            if key.as_str() == Some("jobs") {
                continue;
            }
            for (path, text) in strings(value) {
                match expressions(text, false) {
                    Ok(found) => {
                        for expression in found {
                            for reference in references(&expression, "inputs") {
                                if !inputs.contains(&reference[0]) {
                                    say(format!("{path}: no input `{}`", reference[0]));
                                }
                            }
                        }
                    }
                    Err(problem) => say(format!("{path}: {problem}")),
                }
            }
        }
        for (job_name, job) in &jobs {
            let mut say = |problem: String| say(format!("job {job_name}: {problem}"));
            let needs = names(job.get("needs"));
            for needed in &needs {
                if !jobs.contains_key(needed.as_str()) || needed == job_name {
                    say(format!("needs `{needed}`, which is no other job here"));
                }
            }
            let mut ids = BTreeSet::new();
            match job.get("uses").and_then(Value::as_str) {
                Some(target) => {
                    if job.get("steps").is_some() || job.get("runs-on").is_some() {
                        say("calls a workflow and has steps or a runner of its own".into());
                    }
                    if let Some(file) = target.strip_prefix(CALLED) {
                        match parsed.get(file).map(call_inputs) {
                            // One that is not YAML has been named already.
                            None if files.contains_key(file) => {}
                            None => say(format!("calls {file}, which is not here")),
                            Some(None) => {
                                say(format!("calls {file}, which takes no workflow_call"))
                            }
                            Some(Some(declared)) => {
                                let given = job
                                    .get("with")
                                    .and_then(Value::as_mapping)
                                    .map(|with| with.keys().filter_map(Value::as_str).collect())
                                    .unwrap_or_else(BTreeSet::new);
                                for key in &given {
                                    if !declared.contains_key(key) {
                                        say(format!(
                                            "gives {file} `{key}`, which it does not take"
                                        ));
                                    }
                                }
                                for (key, required) in &declared {
                                    if *required && !given.contains(key) {
                                        say(format!("leaves out `{key}`, which {file} requires"));
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    if job.get("runs-on").is_none() {
                        say("no runner".into());
                    }
                    let steps = job.get("steps").and_then(Value::as_sequence);
                    if steps.is_none_or(|steps| steps.is_empty()) {
                        say("no steps".into());
                    }
                    for (index, step) in steps.into_iter().flatten().enumerate() {
                        if step.get("run").is_some() == step.get("uses").is_some() {
                            say(format!(
                                "step {}: one of `run` and `uses`, not both or neither",
                                index + 1
                            ));
                        }
                        if let Some(id) = step.get("id").and_then(Value::as_str)
                            && !ids.insert(id.to_owned())
                        {
                            say(format!("two steps are `{id}`"));
                        }
                    }
                }
            }
            let matrix = matrix_keys(job);
            for (path, text) in strings(job) {
                let found = match expressions(text, path.ends_with("if")) {
                    Ok(found) => found,
                    Err(problem) => {
                        say(format!("{path}: {problem}"));
                        continue;
                    }
                };
                for expression in found {
                    for reference in references(&expression, "steps") {
                        if !ids.contains(&reference[0]) {
                            say(format!("{path}: no step `{}`", reference[0]));
                        }
                    }
                    for reference in references(&expression, "inputs") {
                        if !inputs.contains(&reference[0]) {
                            say(format!("{path}: no input `{}`", reference[0]));
                        }
                    }
                    for reference in references(&expression, "matrix") {
                        if matrix
                            .as_ref()
                            .is_some_and(|keys| !keys.contains(&reference[0]))
                        {
                            say(format!("{path}: no matrix entry `{}`", reference[0]));
                        }
                    }
                    for reference in references(&expression, "needs") {
                        let needed = &reference[0];
                        if !needs.contains(needed) {
                            say(format!(
                                "{path}: reads `{needed}`, which the job does not need"
                            ));
                            continue;
                        }
                        let (Some("outputs"), Some(output), Some(other)) = (
                            reference.get(1).map(String::as_str),
                            reference.get(2),
                            jobs.get(needed.as_str()),
                        ) else {
                            continue;
                        };
                        let declared = match other.get("uses").and_then(Value::as_str) {
                            // A called workflow's outputs are its `workflow_call` ones.
                            Some(target) => target
                                .strip_prefix(CALLED)
                                .and_then(|file| parsed.get(file))
                                .and_then(|callee| {
                                    callee.get("on")?.get("workflow_call")?.get("outputs")
                                }),
                            None => other.get("outputs"),
                        };
                        if declared.is_none_or(|outputs| outputs.get(output.as_str()).is_none()) {
                            say(format!("{path}: `{needed}` has no output `{output}`"));
                        }
                    }
                }
            }
        }
    }
    wrong
}

/// One name or a list of them.
fn names(value: Option<&Value>) -> BTreeSet<String> {
    match value {
        Some(Value::String(name)) => BTreeSet::from([name.clone()]),
        Some(Value::Sequence(list)) => list
            .iter()
            .filter_map(|name| name.as_str().map(str::to_owned))
            .collect(),
        _ => BTreeSet::new(),
    }
}

/// The inputs a run can have, from a person or from a calling workflow.
fn declared_inputs(workflow: &Value) -> BTreeSet<String> {
    ["workflow_dispatch", "workflow_call"]
        .iter()
        .filter_map(|event| workflow.get("on")?.get(event)?.get("inputs")?.as_mapping())
        .flat_map(|inputs| inputs.keys().filter_map(Value::as_str).map(str::to_owned))
        .collect()
}

/// A callable workflow's inputs and whether each is required; None when it cannot be called.
fn call_inputs(workflow: &Value) -> Option<BTreeMap<&str, bool>> {
    let call = workflow.get("on")?.as_mapping()?.get("workflow_call")?;
    Some(
        call.get("inputs")
            .and_then(Value::as_mapping)
            .into_iter()
            .flatten()
            .filter_map(|(key, input)| {
                Some((
                    key.as_str()?,
                    input.get("required").and_then(Value::as_bool) == Some(true),
                ))
            })
            .collect(),
    )
}

/// The entries a job's matrix has; None when an expression decides them.
fn matrix_keys(job: &Value) -> Option<BTreeSet<String>> {
    let matrix = job.get("strategy")?.get("matrix")?.as_mapping()?;
    let mut keys = BTreeSet::new();
    for (key, value) in matrix {
        match (key.as_str()?, value) {
            ("include", Value::Sequence(entries)) => {
                for entry in entries {
                    keys.extend(
                        entry
                            .as_mapping()?
                            .keys()
                            .filter_map(Value::as_str)
                            .map(str::to_owned),
                    );
                }
            }
            ("include", _) => return None,
            ("exclude", _) => {}
            (key, _) => drop(keys.insert(key.to_owned())),
        }
    }
    Some(keys)
}

/// Every string under `value`, with the keys that lead to it.
fn strings(value: &Value) -> Vec<(String, &str)> {
    fn walk<'a>(value: &'a Value, path: &str, found: &mut Vec<(String, &'a str)>) {
        match value {
            Value::String(text) => found.push((path.to_owned(), text)),
            Value::Sequence(list) => {
                for (index, item) in list.iter().enumerate() {
                    let name = item
                        .get("name")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str);
                    let item_path = match name {
                        Some(name) => format!("{path}[{name}]"),
                        None => format!("{path}[{}]", index + 1),
                    };
                    walk(item, &item_path, found);
                }
            }
            Value::Mapping(entries) => {
                for (key, item) in entries {
                    let key = key.as_str().unwrap_or("?");
                    let separator = if path.is_empty() { "" } else { "." };
                    walk(item, &format!("{path}{separator}{key}"), found);
                }
            }
            _ => {}
        }
    }
    let mut found = Vec::new();
    walk(value, "", &mut found);
    found
}

/// The expressions in `text`: what stands between `${{` and `}}`, or all of a
/// condition written without them.
fn expressions(text: &str, condition: bool) -> Result<Vec<String>, String> {
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("${{") {
        let body = &rest[start + 3..];
        let mut quoted = false;
        let end = body.char_indices().find_map(|(index, character)| {
            if character == '\'' {
                quoted = !quoted;
            }
            (!quoted && body[index..].starts_with("}}")).then_some(index)
        });
        let Some(end) = end else {
            let shown: String = rest[start..].chars().take(40).collect();
            return Err(format!("an expression never closes: {shown}"));
        };
        found.push(body[..end].to_owned());
        rest = &body[end + 2..];
    }
    if condition && found.is_empty() {
        found.push(text.to_owned());
    }
    Ok(found)
}

/// What an expression reads from `context`: `steps.key.outputs.hit` is
/// `["key", "outputs", "hit"]` of `steps`. Words inside its strings are not read.
fn references(expression: &str, context: &str) -> Vec<Vec<String>> {
    let mut quoted = false;
    let plain: String = expression
        .chars()
        .map(|character| {
            if character == '\'' {
                quoted = !quoted;
                ' '
            } else if quoted {
                ' '
            } else {
                character
            }
        })
        .collect();
    plain
        .split(|character: char| !(character.is_ascii_alphanumeric() || "_-.".contains(character)))
        .filter_map(|word| word.strip_prefix(context)?.strip_prefix('.'))
        .map(|rest| rest.split('.').map(str::to_owned).collect::<Vec<_>>())
        .filter(|parts| !parts[0].is_empty())
        .collect()
}

#[test]
fn the_workflows_are_ones_github_accepts() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows");
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(&directory).unwrap() {
        let path = entry.unwrap().path();
        if path
            .extension()
            .is_some_and(|extension| extension == "yml" || extension == "yaml")
        {
            let name = path.file_name().unwrap().to_str().unwrap().to_owned();
            files.insert(name, std::fs::read_to_string(&path).unwrap());
        }
    }
    assert!(files.len() >= 5, "{:?}", files.keys());
    let wrong = problems(&files);
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// A caller and the workflow it calls, with nothing wrong.
const CALLER: &str = r#"
on:
  workflow_dispatch:
    inputs:
      dry_run: { type: boolean, default: true }
jobs:
  plan:
    runs-on: ubuntu-24.04
    outputs:
      systems: ${{ steps.plan.outputs.systems }}
    steps:
      - id: plan
        run: echo 'systems=["linux"]' >> "$GITHUB_OUTPUT"
  build:
    needs: plan
    if: inputs.dry_run && needs.plan.outputs.systems != '[]'
    strategy:
      matrix:
        system: ${{ fromJSON(needs.plan.outputs.systems) }}
        include:
          - system: linux
            runner: ubuntu-24.04
    runs-on: ${{ matrix.runner }}
    steps:
      - id: built
        run: echo "${{ matrix.system }} ${{ format('{0}}}', github.ref) }}"
      - if: steps.built.outcome == 'success'
        uses: actions/upload-artifact@0000000000000000000000000000000000000000
  desktop:
    needs: [plan, build]
    uses: ./.github/workflows/called.yml
    with:
      platforms: ${{ needs.plan.outputs.systems }}
"#;
const CALLEE: &str = r#"
on:
  workflow_call:
    inputs:
      platforms: { type: string, required: true }
      archive: { type: string, default: build }
jobs:
  app:
    runs-on: ubuntu-24.04
    steps:
      - if: inputs.archive != 'artifact'
        run: echo "${{ inputs.platforms }}"
"#;

/// The problems of the pair above after `old` in one of them becomes `new`.
fn after(file: &str, old: &str, new: &str) -> Vec<String> {
    let mut files = BTreeMap::from([
        ("caller.yml".to_owned(), CALLER.to_owned()),
        ("called.yml".to_owned(), CALLEE.to_owned()),
    ]);
    let text = files.get_mut(file).unwrap();
    assert!(text.contains(old), "{old}");
    *text = text.replacen(old, new, 1);
    problems(&files)
}

#[test]
fn each_mistake_github_would_refuse_is_named() {
    assert_eq!(after("caller.yml", "plan", "plan"), Vec::<String>::new());
    let cases = [
        // A plain scalar cannot hold ": ", so the whole file is refused.
        (
            "caller.yml",
            "runs-on: ${{ matrix.runner }}",
            "runs-on: ${{ fromJSON('{\"a\": \"b\"}')[matrix.runner] }}",
            "caller.yml: not YAML",
        ),
        (
            "caller.yml",
            "needs: plan\n",
            "needs: prepare\n",
            "job build: needs `prepare`, which is no other job here",
        ),
        (
            "caller.yml",
            "needs: [plan, build]",
            "needs: [build]",
            "job desktop: with.platforms: reads `plan`, which the job does not need",
        ),
        (
            "caller.yml",
            "needs.plan.outputs.systems != '[]'",
            "needs.plan.outputs.runners != '[]'",
            "job build: if: `plan` has no output `runners`",
        ),
        (
            "caller.yml",
            "inputs.dry_run &&",
            "inputs.dryrun &&",
            "job build: if: no input `dryrun`",
        ),
        (
            "caller.yml",
            "${{ matrix.system }}",
            "${{ matrix.platform }}",
            "no matrix entry `platform`",
        ),
        (
            "caller.yml",
            "steps.built.outcome",
            "steps.build.outcome",
            "job build: steps[2].if: no step `build`",
        ),
        (
            "caller.yml",
            "      - id: built\n",
            "      - id: built\n        uses: actions/checkout@0000000000000000000000000000000000000000\n",
            "job build: step 1: one of `run` and `uses`, not both or neither",
        ),
        (
            "caller.yml",
            "github.ref) }}\"",
            "github.ref) }\"",
            "an expression never closes",
        ),
        (
            "caller.yml",
            "      platforms: ${{ needs.plan.outputs.systems }}",
            "      systems: ${{ needs.plan.outputs.systems }}",
            "job desktop: gives called.yml `systems`, which it does not take",
        ),
        (
            "caller.yml",
            "      platforms: ${{ needs.plan.outputs.systems }}",
            "      archive: artifact",
            "job desktop: leaves out `platforms`, which called.yml requires",
        ),
        (
            "caller.yml",
            "called.yml",
            "missing.yml",
            "job desktop: calls missing.yml, which is not here",
        ),
        (
            "called.yml",
            "  workflow_call:",
            "  workflow_dispatch:",
            "job desktop: calls called.yml, which takes no workflow_call",
        ),
        (
            "called.yml",
            "inputs.archive != 'artifact'",
            "inputs.archives != 'artifact'",
            "called.yml: job app: steps[1].if: no input `archives`",
        ),
    ];
    for (file, old, new, expected) in cases {
        let wrong = after(file, old, new);
        assert!(
            wrong.iter().any(|problem| problem.contains(expected)),
            "{expected}\n{}",
            wrong.join("\n")
        );
    }
}
