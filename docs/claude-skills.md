# Claude Code Skills — kist-drl-g1-workstation

이 레포는 Claude Code에서 자동으로 발견되는 **프로젝트 전용 스킬**을 제공합니다.
Notion 워크스페이스(SYS-REQ / Tasks / Tests / ICD)와 연동하여 컨텍스트 조회와
태스크 자동 생성을 돕습니다.

## 위치 / 구조

```
.claude/
└── skills/
    ├── project-context/SKILL.md    # 읽기 전용 현황 대시보드
    └── task-generator/SKILL.md     # SYS-REQ → Task / Test / ICD 생성
```

스킬은 레포 루트에서 `claude` 실행 시 **자동 로드**됩니다. 별도 설치 단계 없음.

## 사전 요구사항

### 1. Notion MCP 커넥터 연결

두 스킬 모두 Notion DB 읽기/쓰기에 의존하므로 Notion MCP가 설정돼 있어야 합니다.

**Claude Code:**
```bash
claude mcp add notion
```
또는 `~/.claude/mcp_servers.json`에 직접 추가.

**Claude Desktop:**
Settings → Connectors → Notion 연결.

### 2. Notion 워크스페이스 권한

각 스킬의 `Notion DB References` 표에 적힌 DB에 대한 권한:
- `project-context`: **읽기**만 (Meta Data / SYS-REQ / Tasks / Tests / ICD / KIST MBO)
- `task-generator`: **읽기 + 쓰기** (단, SYS-REQ는 읽기만)

권한 부족 시 스킬은 어떤 항목에 접근 못 했는지 사용자에게 알려줍니다.

## 사용법

스킬은 두 가지 방식으로 트리거됩니다.

### A. 명시적 호출 (slash command)

```
/project-context
/task-generator
```

### B. 자연어 매칭

각 스킬의 `description` 필드(YAML frontmatter)에 매칭되는 의도가 감지되면
Claude가 자동으로 스킬을 호출합니다.

| 스킬 | 트리거 예시 |
|---|---|
| `project-context` | "프로젝트 현황 보여줘" / "이번 스프린트는?" / "내 태스크" / "current sprint" |
| `task-generator` | "REQ-31에서 태스크 만들어줘" / "테스트 케이스 작성" / "ICD 등록해줘" |

## 안전 규칙

`task-generator` 사용 시 항상 보장되는 동작:

1. **미리보기 → 사용자 확인 → 생성** — 생성 작업은 사용자가 명시적으로 "확인" 응답을 줘야만 실행됩니다.
2. **SYS-REQ 생성 차단** — SYS-REQ는 PM(SeongYong) 전용이므로 스킬이 생성하지 않습니다.
3. **중복 경고** — REQ에 이미 연결된 Task/Test가 있으면 먼저 표시합니다.
4. **DEPRECATED 차단** — `[DEPRECATED]` 마크된 REQ/ICD에서는 자동 생성을 거부합니다.

`project-context`는 **읽기 전용**이므로 Notion DB를 변경하지 않습니다.

## 스킬 수정 / 추가

### 기존 스킬 수정
`SKILL.md`를 직접 편집한 후 Claude Code 세션을 재시작하면 변경사항이 반영됩니다.

YAML frontmatter (`name`, `description`)는 자연어 트리거 매칭에 사용되므로
변경 시 트리거 동작도 같이 바뀝니다.

### 새 스킬 추가

```
.claude/skills/<skill-name>/SKILL.md
```

최소 형식:
```markdown
---
name: skill-name
description: 언제 이 스킬이 트리거돼야 하는지 한 문장으로 (자연어 매칭에 사용됨).
---

# skill-name

본문(트리거 키워드, 실행 단계, 출력 형식 등).
```

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `/project-context` 명령을 못 찾음 | 세션을 종료하고 레포 루트에서 다시 `claude` 실행. 스킬은 시작 시점에만 로드됨 |
| Notion DB가 비어 보임 | MCP 커넥터가 연결됐는지, 워크스페이스 접근 권한이 있는지 확인 |
| 자연어로 호출했는데 다른 스킬이 떴음 | `description` 필드가 너무 일반적임. 더 구체적인 키워드(`KIST`, `REQ-`)를 포함하도록 수정 |
| `[DEPRECATED]` REQ에서 생성 거부됨 | 의도된 동작. 새 REQ를 PM에게 요청 후 진행 |

## 관련 문서

- [Notion CONV 페이지](https://app.notion.com/p/377b39de7dd780b391f3ceec30226a0e) — 코드 컨벤션 (CONV-001..013, Notion 단일 출처)
- [README.md](../README.md) — 프로젝트 개요 및 Notion DB 링크
