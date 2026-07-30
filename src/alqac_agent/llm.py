from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .config import Settings
from .schemas import (
    AdvocateOpinion,
    CaseTypeClassification,
    DispositionAssessment,
    EvidenceAudit,
    JudgeDecision,
    KeywordExtraction,
    LawSelection,
    SearchPlan,
)

T = TypeVar("T", bound=BaseModel)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


class LegalAgents:
    """Structured-output agents backed by a local vLLM OpenAI-compatible server."""

    def __init__(self, settings: Settings) -> None:
        self.structured_method = settings.llm_structured_method
        self.provider = settings.llm_provider
        if self.provider == "ollama":
            ollama_url = settings.llm_base_url.removesuffix("/v1").rstrip("/")
            self.chat = ChatOllama(
                model=settings.llm_model,
                base_url=ollama_url,
                temperature=settings.llm_temperature,
                reasoning=False,
                num_predict=settings.llm_max_tokens,
                client_kwargs={"timeout": settings.llm_timeout_seconds},
            )
        elif self.provider == "groq":
            self.chat = ChatOpenAI(
                model=settings.llm_model,
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.groq_api_key,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
                max_retries=2,
            )
        else:
            self.chat = ChatOpenAI(
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
                max_retries=2,
            )

    def _invoke(self, schema: type[T], system: str, payload: object) -> T:
        system = (
            "Luôn phân tích và viết nội dung bằng tiếng Việt. Chỉ giữ nguyên tiếng Anh "
            "cho identifier hoặc nhãn bắt buộc như A_WIN/B_WIN.\n\n" + system
        )
        if self.provider == "ollama":
            runnable = self.chat.with_structured_output(schema, method="json_schema")
            result = runnable.invoke(
                [SystemMessage(content=system), HumanMessage(content=_json(payload))]
            )
            return result if isinstance(result, schema) else schema.model_validate(result)

        if self.structured_method == "json_schema":
            runnable = self.chat.with_structured_output(schema, method="json_schema")
            result = runnable.invoke(
                [SystemMessage(content=system), HumanMessage(content=_json(payload))]
            )
            return result if isinstance(result, schema) else schema.model_validate(result)

        schema_json = _json(schema.model_json_schema())
        instruction = f"""{system}

QUAN TRỌNG: Chỉ trả về đúng MỘT JSON object, không markdown, không giải thích ngoài
JSON. JSON phải tuân thủ chính xác schema sau; không đổi tên field và không thêm field:
{schema_json}"""
        messages = [SystemMessage(content=instruction), HumanMessage(content=_json(payload))]
        last_error: Exception | None = None
        for attempt in range(3):
            response = self.chat.invoke(
                messages,
                response_format={"type": "json_object"},
            )
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            text = str(content).strip()
            fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
            if fenced:
                text = fenced.group(1)
            try:
                return schema.model_validate(json.loads(text))
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                messages.extend(
                    [
                        response,
                        HumanMessage(
                            content=(
                                "Output trước sai schema. Hãy sửa và chỉ trả JSON đúng schema. "
                                f"Validation error: {error}"
                            )
                        ),
                    ]
                )
        raise ValueError(f"LLM không trả được {schema.__name__} hợp lệ: {last_error}")

    def plan(self, case_query: str) -> SearchPlan:
        return self._invoke(
            SearchPlan,
            """Bạn là agent lập kế hoạch truy hồi án lệ Việt Nam. Từ case_query, xác định
yêu cầu chính cần chấm, vấn đề pháp lý quyết định, và các truy vấn BM25 ngắn để tìm
trong đúng hồ sơ vụ án. Phải ưu tiên truy vấn tìm phần NHẬN ĐỊNH, QUYẾT ĐỊNH/TUYÊN
XỬ, câu 'chấp nhận/không chấp nhận yêu cầu khởi kiện', sau đó mới đến tình tiết.
Mỗi case_queries phải là một cụm từ khóa tiếng Việt có khả năng xuất hiện nguyên
văn trong bản án. Không viết câu lệnh kiểu 'tìm', 'search', 'find', 'locate'; không
yêu cầu nguồn bên ngoài hồ sơ. Không dự đoán kết quả. Trả đúng schema JSON.""",
            {"case_query": case_query},
        )

    def audit(
        self, case_query: str, evidence: list[dict[str, object]]
    ) -> EvidenceAudit:
        return self._invoke(
            EvidenceAudit,
            """Bạn kiểm tra độ đủ của chứng cứ vụ án. sufficient=true chỉ khi các đoạn
đã cho đủ để xác định toàn bộ, một phần trên 50%, một phần không quá 50%, hay bác
toàn bộ yêu cầu chính. Nếu thiếu, tạo tối đa 2 truy vấn BM25 thật cụ thể để tìm đúng
phần còn thiếu; không lặp truy vấn cũ. Trả đúng schema JSON.""",
            {"case_query": case_query, "evidence": evidence},
        )

    def extract_disposition(
        self,
        case_query: str,
        evidence: list[dict[str, object]],
        laws: list[dict[str, object]] | None = None,
    ) -> DispositionAssessment:
        compact_laws = [
            {**item, "content": str(item.get("content", ""))[:500]}
            for item in (laws or [])[:5]
        ]
        return self._invoke(
            DispositionAssessment,
            """Bạn là agent trích xuất PHÁN QUYẾT, không phải luật sư tranh luận.
Chỉ xét yêu cầu chính được mô tả trong case_query và ưu tiên case_evidence. Law
context chỉ dùng để hiểu quan hệ pháp luật, căn cứ và phạm vi yêu cầu; không dùng
law context để đoán kết quả nếu case_evidence không cho biết Tòa thực tế xử thế
nào. Phân biệt quyết định của Tòa với lời khai đương sự hoặc đề nghị của Viện kiểm
sát. Nếu có cả quyết định sơ thẩm và phúc thẩm, ưu tiên quyết định cuối cùng.

So sánh cụ thể phần nguyên đơn yêu cầu với phần Tòa thực tế chấp nhận, gồm số tiền,
diện tích, tài sản và các yêu cầu thành phần. ALL chỉ khi chấp nhận đúng 100%;
MAJORITY khi trên 50% nhưng dưới 100%; MINORITY khi trên 0% đến 50%; NONE khi bác
toàn bộ. Nếu evidence không đủ để so sánh, claim_scope=UNCLEAR và tạo tối đa 2
followup_queries bằng cụm từ tiếng Việt có thể xuất hiện trong phần QUYẾT ĐỊNH.

decisive_chunk_ids chỉ được chứa ID có trong evidence. Không dự đoán theo luật hoặc
theo bên nào có vẻ hợp lý. Trả đúng schema JSON.""",
            {
                "case_query": case_query,
                "case_evidence": evidence,
                "law_context": compact_laws,
            },
        )

    def select_laws(
        self,
        case_query: str,
        case_evidence: list[dict[str, object]],
        candidates: list[dict[str, object]],
    ) -> LawSelection:
        compact = [
            {**item, "content": str(item["content"])[:600]}
            for item in candidates[:10]
        ]
        return self._invoke(
            LawSelection,
            """Bạn là chuyên gia truy hồi điều luật. Chỉ chọn các cặp law_id/aid có
trong candidates và trực tiếp chi phối yêu cầu chính hoặc căn cứ tố tụng thiết yếu.
selected là danh sách object {law_id, aid}; không bịa identifier, không chọn tràn lan.
Trả đúng schema JSON.""",
            {
                "case_query": case_query,
                "case_evidence": case_evidence,
                "candidates": compact,
            },
        )

    def _advocate(
        self,
        side: str,
        case_query: str,
        evidence: list[dict[str, object]],
        laws: list[dict[str, object]],
    ) -> AdvocateOpinion:
        role = "nguyên đơn (A)" if side == "A" else "bị đơn (B)"
        return self._invoke(
            AdvocateOpinion,
            f"""Bạn là luật sư bảo vệ {role}. Phân tích trung thực hồ sơ và điều luật,
nhưng trình bày lập luận mạnh nhất cho thân chủ. Nhãn có nghĩa: A_WIN=chấp nhận
toàn bộ; PARTIAL_A_WIN=chấp nhận một phần >50%; PARTIAL_B_WIN=chấp nhận một
phần <=50%; B_WIN=bác toàn bộ. Chỉ dựa trên nội dung được cung cấp. side phải là
{side}. proposed_label và confidence phải được ghi trước phần analysis. Analysis
ngắn gọn, tối đa 600 ký tự. Trả đúng schema JSON.""",
            {"case_query": case_query, "case_evidence": evidence, "laws": laws},
        )

    def debate(
        self,
        case_query: str,
        evidence: list[dict[str, object]],
        laws: list[dict[str, object]],
    ) -> tuple[AdvocateOpinion, AdvocateOpinion]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            a_future = executor.submit(self._advocate, "A", case_query, evidence, laws)
            b_future = executor.submit(self._advocate, "B", case_query, evidence, laws)
            return a_future.result(), b_future.result()

    def judge(
        self,
        case_query: str,
        evidence: list[dict[str, object]],
        laws: list[dict[str, object]],
        examples: list[dict[str, object]],
        advocate_a: dict[str, object],
        advocate_b: dict[str, object],
    ) -> JudgeDecision:
        return self._invoke(
            JudgeDecision,
            """Bạn là thẩm phán phản biện cuối cùng. Ưu tiên câu chữ QUYẾT ĐỊNH và
mức chấp nhận yêu cầu chính trong case_query, không đánh đồng 'có thắng một phần'
với A_WIN. Quy tắc nhãn: chấp nhận 100%=A_WIN; chấp nhận >50% nhưng <100%=
PARTIAL_A_WIN; chấp nhận >0% và <=50%=PARTIAL_B_WIN; 0%=B_WIN. Ví dụ public
chỉ giúp hiệu chỉnh cách gán nhãn, không phải chứng cứ cho vụ hiện tại. Chỉ chọn
law_id/aid đã có trong laws. Ghi prediction trước explanation; explanation tối đa
800 ký tự. Trả đúng schema JSON.""",
            {
                "case_query": case_query,
                "case_evidence": evidence,
                "laws": laws,
                "nearest_public_examples": examples,
                "plaintiff_advocate": advocate_a,
                "defendant_advocate": advocate_b,
            },
        )

    def classify_case_type(self, case_query: str) -> CaseTypeClassification:
        case_types = [
            {"case_type_id": "land_use_dispute", "case_type_text": "Tranh chấp quyền sử dụng đất, ranh giới, lối đi, lấn chiếm"},
            {"case_type_id": "compensation_damage", "case_type_text": "Tranh chấp bồi thường thiệt hại ngoài hợp đồng"},
            {"case_type_id": "inheritance", "case_type_text": "Tranh chấp di sản thừa kế"},
            {"case_type_id": "credit_contract", "case_type_text": "Tranh chấp hợp đồng tín dụng"},
            {"case_type_id": "land_transfer_contract", "case_type_text": "Tranh chấp hợp đồng chuyển nhượng quyền sử dụng đất"},
            {"case_type_id": "loan_contract", "case_type_text": "Tranh chấp hợp đồng vay mượn tài sản"},
            {"case_type_id": "sale_goods_debt", "case_type_text": "Tranh chấp hợp đồng mua bán hàng hóa, công nợ"},
            {"case_type_id": "service_labor_training", "case_type_text": "Tranh chấp hợp đồng dịch vụ, lao động, đào tạo"},
            {"case_type_id": "hui_dispute", "case_type_text": "Tranh chấp hụi"},
            {"case_type_id": "cooperation_contract", "case_type_text": "Tranh chấp hợp đồng hợp tác"},
            {"case_type_id": "property_claim", "case_type_text": "Tranh chấp quyền sở hữu tài sản"},
        ]
        return self._invoke(
            CaseTypeClassification,
            """Bạn là một chuyên gia pháp lý tại Việt Nam, chuyên thực hiện công việc phân loại các vụ án theo từng loại vụ việc có trong danh sách "case_types" được cung cấp.

Bạn sẽ được đọc một nội dung tóm tắt vụ việc gọi là "case_query" và sẽ suy luận để phân loại vào từng loại vụ việc trong "case_types" được cung cấp.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong case_query và danh sách case_types.
2. Chỉ được chọn case_type_id và case_type_text xuất hiện trong danh sách case_types.
3. Nếu case_query có nhiều dấu hiệu, chọn loại vụ việc chính làm primary_case_type.
4. Trả về duy nhất một JSON hợp lệ, không thêm văn bản ngoài JSON.

Ngoài ra, dựa trên case_query, xác định:
- predicted_difficulty: easy/medium/hard dựa trên số bên, loại tranh chấp, độ phức tạp.
- key_facts: các sự kiện pháp lý quan trọng nhất (tối đa 6).""",
            {"case_types": case_types, "case_query": case_query},
        )

    def extract_keywords(self, case_query: str) -> KeywordExtraction:
        return self._invoke(
            KeywordExtraction,
            """Bạn là một chuyên gia về văn học, cấu trúc câu, dạng từ ngữ, động từ, danh từ, tính từ, tên riêng, con số, với ngôn ngữ tiếng Việt.
Công việc của bạn là từ một văn bản đầu vào, bạn rút trích TẤT CẢ những gì có trong văn bản, bao gồm:
- Họ tên người, tên tổ chức/cơ quan/doanh nghiệp.
- Danh từ hoặc cụm danh từ.
- Động từ hoặc cụm động từ.
- Tính từ hoặc cụm tính từ.
- Từ chỉ địa điểm, địa danh hoặc tên địa danh.
- Các con số.
Quy tắc bắt buộc:
1. Tất cả những gì bạn đã rút trích đều thuộc trong văn bản đầu vào, không được bịa đặt.
2. Không rút trích các cụm trong câu cuối liên quan về "Agent dự đoán", "theo bạn", "Thắng kiện".
3. Mỗi keyword phải được định danh bằng một mã duy nhất theo đúng loại:
   - name1, name2, ... cho họ tên người hoặc tên tổ chức.
   - noun1, noun2, ... cho danh từ hoặc cụm danh từ.
   - verb1, verb2, ... cho động từ hoặc cụm động từ.
   - adjective1, adjective2, ... cho tính từ.
   - location1, location2, ... cho địa điểm.
   - number1, number2, ... cho con số, số tiền, diện tích.
4. Trả về duy nhất một JSON hợp lệ, không thêm văn bản ngoài JSON.
5. Phải rút trích được nhiều nhất có thể.""",
            {"case_query": case_query},
        )
