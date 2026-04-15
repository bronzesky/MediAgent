"""
C3: PathoGraph诊断流程引导的可验证Agent推理

核心功能:
1. 三阶段结构化推理 (PathoGraph Diagnosis Graph)
2. Graph-RAG历史病例检索 (WL kernel)
3. 图拓扑硬约束验证器 (区别于WSI-Agents的软约束)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ReasoningStage:
    """推理阶段输出"""
    stage: str  # "preliminary", "further", "final"
    diagnosis_list: List[str]
    phenotypes_to_query: List[str]
    reasoning: str
    evidence_hyperedges: List[int]  # 引用的超边ID
    confidence: float


@dataclass
class VerificationResult:
    """验证结果"""
    is_valid: bool
    similarity_score: float
    threshold: float
    matched_hyperedges: List[int]
    feedback: str


class GraphTopologyVerifier(nn.Module):
    """
    图拓扑硬约束验证器。
    
    原理:
    - LLM生成的描述 → CONCH text encoder → 文本embedding
    - 在超图中检索最近邻超边 (使用C2对齐后的表示)
    - 相似度 < τ → 判定幻觉，返回重生成
    
    与WSI-Agents的区别:
    - WSI-Agents: LLM内部一致性检查 (软约束)
    - 本工作: 图拓扑硬约束 (外部结构验证)
    """
    
    def __init__(
        self,
        text_encoder: Any,  # CONCH text encoder
        aligner: Any,  # C2 PathoGraphAligner
        threshold: float = 0.5,
        top_k: int = 3,
    ):
        """
        Args:
            text_encoder: CONCH文本编码器
            aligner: C2对齐模块 (用于获取对齐后的超边表示)
            threshold: 相似度阈值 (低于此值判定为幻觉)
            top_k: 检索top-k个最相似超边
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.aligner = aligner
        self.threshold = threshold
        self.top_k = top_k
    
    @torch.no_grad()
    def verify(
        self,
        description: str,
        hyperedge_features: Tensor,
        hyperedge_ids: Optional[List[int]] = None,
    ) -> VerificationResult:
        """
        验证LLM生成的描述是否有图拓扑支持。
        
        Args:
            description: LLM生成的表型描述
            hyperedge_features: (K, d_h) 超边特征
            hyperedge_ids: 超边ID列表 (用于追溯)
        
        Returns:
            VerificationResult: 验证结果
        """
        # 编码文本描述
        text_embedding = self.text_encoder.encode_text([description])  # (1, d_t)
        
        # 计算与超边的相似度
        similarity = self.aligner.compute_text_similarity(
            hyperedge_features=hyperedge_features,
            text_features=text_embedding,
        )  # (K, 1)
        
        similarity = similarity.squeeze(-1)  # (K,)
        
        # 找到top-k最相似的超边
        top_k = min(self.top_k, similarity.size(0))
        top_scores, top_indices = torch.topk(similarity, k=top_k)
        
        max_score = top_scores[0].item()
        
        # 判断是否通过验证
        is_valid = max_score >= self.threshold
        
        # 构建反馈
        if is_valid:
            matched_ids = [hyperedge_ids[i] if hyperedge_ids else i for i in top_indices.tolist()]
            feedback = f"Verified: Found supporting evidence in hyperedges {matched_ids[:3]} (similarity: {max_score:.3f})"
        else:
            feedback = (
                f"Hallucination detected: No hyperedge matches description '{description[:50]}...' "
                f"(max similarity: {max_score:.3f} < threshold: {self.threshold}). "
                f"Please regenerate based on actual graph structure."
            )
        
        return VerificationResult(
            is_valid=is_valid,
            similarity_score=max_score,
            threshold=self.threshold,
            matched_hyperedges=top_indices.tolist() if is_valid else [],
            feedback=feedback,
        )
    
    def set_threshold(self, threshold: float):
        """动态调整阈值"""
        self.threshold = threshold


class ThreeStageReasoner:
    """
    三阶段结构化推理器 (PathoGraph Diagnosis Graph)。
    
    阶段1: Preliminary Diagnosis
    - 输入: 组织节点全局特征 + Graph-RAG检索结果
    - 输出: 鉴别诊断列表 + 需查询的表型列表
    
    阶段2: Further Diagnosis
    - 输入: 阶段1指定的目标超边细节 + PathoGraph量化参数
    - 输出: 缩小的诊断可能性 + 关键特征量化描述
    
    阶段3: Final Diagnosis
    - 输入: 阶段1+2全部中间输出
    - 输出: 最终亚型分类 + 结构化报告 + 完整推理链
    """
    
    def __init__(
        self,
        llm_client: Any,  # Gemini 2.5 Pro API client
        verifier: GraphTopologyVerifier,
        rag_retriever: Any,  # Graph-RAG检索器
        max_retries: int = 3,
    ):
        """
        Args:
            llm_client: LLM API客户端
            verifier: 图拓扑验证器
            rag_retriever: Graph-RAG检索器
            max_retries: 验证失败时的最大重试次数
        """
        self.llm = llm_client
        self.verifier = verifier
        self.rag = rag_retriever
        self.max_retries = max_retries
    
    def stage1_preliminary(
        self,
        tissue_features: Tensor,
        hyperedge_features: Tensor,
        case_id: str,
    ) -> ReasoningStage:
        """
        阶段1: 初步诊断。
        
        Args:
            tissue_features: (M, d_t) 组织节点特征
            hyperedge_features: (K, d_h) 超边特征
            case_id: 病例ID
        
        Returns:
            ReasoningStage: 阶段1输出
        """
        # Graph-RAG检索相似病例
        similar_cases = self.rag.retrieve(tissue_features, top_k=3)
        
        # 构建prompt
        prompt = self._build_stage1_prompt(
            tissue_features=tissue_features,
            similar_cases=similar_cases,
            case_id=case_id,
        )
        
        # LLM推理 + 验证循环
        for attempt in range(self.max_retries):
            response = self.llm.generate(prompt)
            
            # 解析输出
            parsed = self._parse_stage1_response(response)
            
            # 验证每个提到的表型
            all_valid = True
            feedback_list = []
            
            for phenotype_desc in parsed['phenotypes_mentioned']:
                verification = self.verifier.verify(
                    description=phenotype_desc,
                    hyperedge_features=hyperedge_features,
                )
                
                if not verification.is_valid:
                    all_valid = False
                    feedback_list.append(verification.feedback)
            
            if all_valid:
                return ReasoningStage(
                    stage="preliminary",
                    diagnosis_list=parsed['diagnosis_list'],
                    phenotypes_to_query=parsed['phenotypes_to_query'],
                    reasoning=parsed['reasoning'],
                    evidence_hyperedges=parsed['evidence_hyperedges'],
                    confidence=parsed['confidence'],
                )
            else:
                # 添加反馈到prompt，重新生成
                feedback_str = "\n".join(feedback_list)
                prompt += f"\n\n[Verification Feedback]:\n{feedback_str}\nPlease regenerate based on actual graph structure."
        
        # 达到最大重试次数，返回最后一次结果并标记低置信度
        return ReasoningStage(
            stage="preliminary",
            diagnosis_list=parsed['diagnosis_list'],
            phenotypes_to_query=parsed['phenotypes_to_query'],
            reasoning=parsed['reasoning'] + " [Warning: Verification failed]",
            evidence_hyperedges=[],
            confidence=0.3,
        )
    
    def stage2_further(
        self,
        stage1_output: ReasoningStage,
        target_hyperedge_features: Tensor,
        pathograph_params: Dict[str, Any],
    ) -> ReasoningStage:
        """
        阶段2: 进一步诊断。
        
        Args:
            stage1_output: 阶段1输出
            target_hyperedge_features: 阶段1指定的目标超边特征
            pathograph_params: PathoGraph量化参数 (如核大小、形状等)
        
        Returns:
            ReasoningStage: 阶段2输出
        """
        prompt = self._build_stage2_prompt(
            stage1_output=stage1_output,
            pathograph_params=pathograph_params,
        )
        
        # 类似阶段1的验证循环
        for attempt in range(self.max_retries):
            response = self.llm.generate(prompt)
            parsed = self._parse_stage2_response(response)
            
            # 验证
            verification = self.verifier.verify(
                description=parsed['feature_description'],
                hyperedge_features=target_hyperedge_features,
            )
            
            if verification.is_valid:
                return ReasoningStage(
                    stage="further",
                    diagnosis_list=parsed['refined_diagnosis_list'],
                    phenotypes_to_query=[],
                    reasoning=parsed['reasoning'],
                    evidence_hyperedges=verification.matched_hyperedges,
                    confidence=parsed['confidence'],
                )
            else:
                prompt += f"\n\n[Verification Feedback]:\n{verification.feedback}"
        
        return ReasoningStage(
            stage="further",
            diagnosis_list=parsed['refined_diagnosis_list'],
            phenotypes_to_query=[],
            reasoning=parsed['reasoning'] + " [Warning: Verification failed]",
            evidence_hyperedges=[],
            confidence=0.3,
        )
    
    def stage3_final(
        self,
        stage1_output: ReasoningStage,
        stage2_output: ReasoningStage,
        all_hyperedge_features: Tensor,
    ) -> ReasoningStage:
        """
        阶段3: 最终诊断。
        
        Args:
            stage1_output: 阶段1输出
            stage2_output: 阶段2输出
            all_hyperedge_features: 所有超边特征
        
        Returns:
            ReasoningStage: 阶段3输出 (最终诊断)
        """
        prompt = self._build_stage3_prompt(
            stage1_output=stage1_output,
            stage2_output=stage2_output,
        )
        
        for attempt in range(self.max_retries):
            response = self.llm.generate(prompt)
            parsed = self._parse_stage3_response(response)
            
            # 验证最终报告
            verification = self.verifier.verify(
                description=parsed['final_report'],
                hyperedge_features=all_hyperedge_features,
            )
            
            if verification.is_valid:
                return ReasoningStage(
                    stage="final",
                    diagnosis_list=[parsed['final_diagnosis']],
                    phenotypes_to_query=[],
                    reasoning=parsed['complete_reasoning_chain'],
                    evidence_hyperedges=verification.matched_hyperedges,
                    confidence=parsed['confidence'],
                )
            else:
                prompt += f"\n\n[Verification Feedback]:\n{verification.feedback}"
        
        return ReasoningStage(
            stage="final",
            diagnosis_list=[parsed['final_diagnosis']],
            phenotypes_to_query=[],
            reasoning=parsed['complete_reasoning_chain'] + " [Warning: Verification failed]",
            evidence_hyperedges=[],
            confidence=0.3,
        )
    
    def _build_stage1_prompt(self, tissue_features, similar_cases, case_id) -> str:
        """构建阶段1 prompt"""
        prompt = f"""
[Task]: Preliminary Diagnosis for Case {case_id}

[Context]:
You are a pathologist analyzing a breast cancer WSI. Based on tissue-level features and similar historical cases, provide a preliminary differential diagnosis.

[Similar Cases from Graph-RAG]:
{self._format_similar_cases(similar_cases)}

[Instructions]:
1. List 2-3 most likely diagnoses (from BRACS categories: Normal, PB, UDH, FEA, ADH, DCIS, IC)
2. Identify specific phenotypes that need further examination
3. Provide reasoning based on tissue architecture
4. Reference specific hyperedge IDs as evidence

[Output Format]:
{{
  "diagnosis_list": ["diagnosis1", "diagnosis2"],
  "phenotypes_to_query": ["phenotype1", "phenotype2"],
  "reasoning": "...",
  "evidence_hyperedges": [1, 5, 8],
  "confidence": 0.7
}}
"""
        return prompt
    
    def _build_stage2_prompt(self, stage1_output, pathograph_params) -> str:
        """构建阶段2 prompt"""
        prompt = f"""
[Task]: Further Diagnosis

[Stage 1 Output]:
{json.dumps(stage1_output.__dict__, indent=2)}

[PathoGraph Quantitative Parameters]:
{json.dumps(pathograph_params, indent=2)}

[Instructions]:
1. Examine the phenotypes identified in Stage 1
2. Use PathoGraph parameters to quantify key features
3. Narrow down the differential diagnosis
4. Provide detailed feature descriptions

[Output Format]:
{{
  "refined_diagnosis_list": ["diagnosis1"],
  "feature_description": "...",
  "reasoning": "...",
  "confidence": 0.85
}}
"""
        return prompt
    
    def _build_stage3_prompt(self, stage1_output, stage2_output) -> str:
        """构建阶段3 prompt"""
        prompt = f"""
[Task]: Final Diagnosis and Report Generation

[Stage 1 Output]:
{json.dumps(stage1_output.__dict__, indent=2)}

[Stage 2 Output]:
{json.dumps(stage2_output.__dict__, indent=2)}

[Instructions]:
1. Synthesize findings from both stages
2. Provide final diagnosis with confidence
3. Generate structured pathology report
4. Include complete reasoning chain with evidence

[Output Format]:
{{
  "final_diagnosis": "...",
  "final_report": "...",
  "complete_reasoning_chain": "...",
  "confidence": 0.9
}}
"""
        return prompt
    
    def _format_similar_cases(self, similar_cases) -> str:
        """格式化相似病例"""
        lines = []
        for i, case in enumerate(similar_cases, 1):
            lines.append(f"Case {i}: {case['diagnosis']} (similarity: {case['score']:.3f})")
        return "\n".join(lines)
    
    def _parse_stage1_response(self, response: str) -> Dict[str, Any]:
        """解析阶段1响应"""
        # 简化版，实际需要robust的JSON解析
        try:
            return json.loads(response)
        except:
            return {
                'diagnosis_list': [],
                'phenotypes_to_query': [],
                'phenotypes_mentioned': [],
                'reasoning': response,
                'evidence_hyperedges': [],
                'confidence': 0.5,
            }
    
    def _parse_stage2_response(self, response: str) -> Dict[str, Any]:
        """解析阶段2响应"""
        try:
            return json.loads(response)
        except:
            return {
                'refined_diagnosis_list': [],
                'feature_description': response,
                'reasoning': response,
                'confidence': 0.5,
            }
    
    def _parse_stage3_response(self, response: str) -> Dict[str, Any]:
        """解析阶段3响应"""
        try:
            return json.loads(response)
        except:
            return {
                'final_diagnosis': 'Unknown',
                'final_report': response,
                'complete_reasoning_chain': response,
                'confidence': 0.5,
            }


def run_full_reasoning_pipeline(
    reasoner: ThreeStageReasoner,
    tissue_features: Tensor,
    hyperedge_features: Tensor,
    pathograph_params: Dict[str, Any],
    case_id: str,
) -> Dict[str, ReasoningStage]:
    """
    运行完整的三阶段推理pipeline。
    
    Args:
        reasoner: 三阶段推理器
        tissue_features: 组织节点特征
        hyperedge_features: 超边特征
        pathograph_params: PathoGraph参数
        case_id: 病例ID
    
    Returns:
        Dict包含三个阶段的输出
    """
    # 阶段1
    stage1 = reasoner.stage1_preliminary(
        tissue_features=tissue_features,
        hyperedge_features=hyperedge_features,
        case_id=case_id,
    )
    
    # 阶段2 (基于阶段1指定的超边)
    target_hyperedges = stage1.evidence_hyperedges
    target_features = hyperedge_features[target_hyperedges] if target_hyperedges else hyperedge_features
    
    stage2 = reasoner.stage2_further(
        stage1_output=stage1,
        target_hyperedge_features=target_features,
        pathograph_params=pathograph_params,
    )
    
    # 阶段3
    stage3 = reasoner.stage3_final(
        stage1_output=stage1,
        stage2_output=stage2,
        all_hyperedge_features=hyperedge_features,
    )
    
    return {
        'stage1': stage1,
        'stage2': stage2,
        'stage3': stage3,
    }
