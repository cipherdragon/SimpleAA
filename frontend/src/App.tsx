import { useMemo, useState } from "react";
import { getSimilarity } from "./apiHandler";

function App() {
	const [similarityScore, setSimilarityScore] = useState<string>("N/A");
	
	const similarityScoreColor = useMemo(() => {
		const score = +similarityScore;
		if (isNaN(score)) return "slate";

		if (score < 0.5) return "red";
		return "green";
	}, [similarityScore]);

	const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();

		const formData = new FormData(e.target as HTMLFormElement);
		const file1 = formData.get('text_1') as File;
		const file2 = formData.get('text_2') as File;
		const lang = formData.get('text_lang') as string;

		let text1: string | undefined;
		let text2: string | undefined;

		const readText = (file: File) => new Promise<string>((resolve, reject) => {
			const reader = new FileReader();
			reader.onload = e => resolve(e.target?.result as string);
			reader.onerror = e => reject(e.target?.error);
			reader.readAsText(file);
		})

		try {
			text1 = await readText(file1);
			text2 = await readText(file2);
		} catch (error) {
			return alert('Error reading file');
		}

		setSimilarityScore("Calculating...");
		getSimilarity(lang, text1, text2)
			.then(similarity => setSimilarityScore(`${similarity}`))
			.catch(err => {
				setSimilarityScore("Failed to calculate");
				console.log(err);
			});
	};

	return (
		<main className='w-full h-screen bg-slate-200 box-border p-[1px] flex flex-col items-center justify-center gap-[30px]'>
			<h1 className="text-[22px] font-bold text-slate-900 mt-[10px]">
				Simple Authorship Attribution
			</h1>
			<div>
				<form className="grid grid-cols-[auto_200px] gap-2 items-center bg-slate-50 p-3 rounded-md gap-x-[50px] shadow-lg pt-[30px]" onSubmit={handleSubmit}>
					<label htmlFor="text_1" className="text-[14px] font-bold text-slate-800">
						Upload anonymously authored text
					</label>
					<input type="file" name="text_1" id="text_1" className="text-[14px] text-slate-600" />

					<label htmlFor="text_2" className="text-[14px] font-bold text-slate-800">
						Upload suspected author's text
					</label>
					<input type="file" name="text_2" id="text_2" className="text-[14px] text-slate-600"/>

					<label htmlFor="text_lang" className="text-[14px] font-bold text-slate-800">
						Select input language
					</label>
					<select name="text_lang" id="text_lang" className="p-1 text-center">
						<option value="en">English</option>
						<option value="si">Singlish</option>
					</select>

					<button type="submit" className="bg-blue-600 text-slate-50 font-bold p-2 rounded-md col-span-2 mt-[30px]">
						Submit
					</button>
				</form>	
				<div className="mt-[30px] bg-slate-50 rounded-md p-3 shadow-lg grid grid-cols-[auto_200px] gap-x-[50px] text-slate-800 font-bold">
					<span className="">
						Text similarity score : 
					</span> 
					<span className={`text-center ${similarityScoreColor === "slate" ? "text-slate-800" : (similarityScoreColor === "red" ? "text-red-600" : "text-green-600")}`}>
						{isNaN(+similarityScore) ? similarityScore : `${Math.round(+similarityScore * 100)}%`}
					</span>
				</div>
			</div>
			<div></div>
		</main>
	)
}

export default App
