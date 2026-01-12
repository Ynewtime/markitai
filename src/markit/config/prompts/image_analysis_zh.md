分析此图像并返回 JSON 格式的响应。**alt_text 和 detailed_description 必须使用中文**。

## 返回字段说明

1. **alt_text** (中文): 简短的图像描述（1句话，最多50个汉字）
2. **detailed_description** (中文): 图像内容的详细描述（2-5句话）
3. **detected_text**: 图像中可见的文本内容
   - 提取有意义的文本，按阅读顺序排列
   - **忽略**: OCR乱码、装饰性水印、无意义单字符
   - 如果没有有意义的文本，返回 null
4. **image_type**: 图像类型分类
   - diagram | chart | graph | table | screenshot | photo | illustration | logo | icon | formula | code | other
5. **knowledge_meta** (可选): 知识图谱元数据
   - **entities**: 实体列表（人名、组织、产品、技术术语）
   - **relationships**: 实体间关系（格式: "实体A -> 关系 -> 实体B"）
   - **topics**: 主题标签
   - **domain**: 领域分类（技术、商业、学术、医疗等）

## 响应格式要求
- 直接输出有效 JSON 对象，以 { 开头，以 } 结尾
- 不要使用 markdown 代码块（不要使用 ```json 或 ```）
- 不要在 JSON 前后添加任何说明文字

示例输出:
{"alt_text": "展示微服务架构的系统设计图", "detailed_description": "该图展示了一个典型的微服务架构，包含API网关、用户服务、订单服务三个核心组件。", "detected_text": "API Gateway, User Service", "image_type": "diagram", "knowledge_meta": {"entities": ["API网关", "用户服务"], "relationships": ["用户服务 -> 调用 -> API网关"], "topics": ["微服务", "系统架构"], "domain": "技术"}}
